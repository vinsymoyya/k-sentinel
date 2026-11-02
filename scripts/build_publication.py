"""Build the paper, research brief and figures from frozen released results.

Requires reportlab, Pillow, numpy, pypdf and pdftoppm on PATH.
"""
import html
import json
import re
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, String, Line, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               Image, Preformatted, KeepTogether)
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/"assets"; ASSETS.mkdir(exist_ok=True)
OUTPUT=ROOT/"output/pdf"; OUTPUT.mkdir(parents=True,exist_ok=True)
SUMMARY=json.loads((ROOT/"results/main/summary.json").read_text())
G={(g["model"],g["case"],g["budget"]):g for g in SUMMARY["groups"]}
PALETTE=[colors.HexColor(c) for c in ("#087f8c","#db6e24","#6355a5","#334155")]


def group(model,case,budget=4): return G[(model,case,budget)]
def rate(g): return f"{100*g['rejection_rate']:.1f}%"
def count(g): return str(g["flagged"])


def chart_png(drawing,name):
    renderSVG.drawToFile(drawing,str(ASSETS/f"{name}.svg"))
    renderPDF.drawToFile(drawing,str(ASSETS/f"{name}.pdf"))
    subprocess.run(["pdftoppm","-scale-to","1800","-singlefile","-png",
                    str(ASSETS/f"{name}.pdf"),str(ASSETS/name)],check=True,capture_output=True)


def detection_figure():
    d=Drawing(760,375)
    d.add(Rect(0,0,760,375,fillColor=colors.white,strokeColor=None))
    d.add(String(18,352,"Hidden measurement audits: detection and blind spots",fontName="Helvetica-Bold",fontSize=16))
    plot=LinePlot(); plot.x=65; plot.y=86; plot.width=650; plot.height=230
    series=[("oracle","insertion","Oracle insertion"),("cnn","insertion","CNN insertion"),
            ("oracle","single_line","Oracle single-line"),("cnn","audit_aware","CNN audit-aware")]
    plot.data=[[(b,100*group(m,c,b)["rejection_rate"]) for b in (1,2,4,8)] for m,c,_ in series]
    plot.xValueAxis.valueMin=0; plot.xValueAxis.valueMax=8; plot.xValueAxis.valueStep=2
    plot.yValueAxis.valueMin=0; plot.yValueAxis.valueMax=100; plot.yValueAxis.valueStep=20
    plot.xValueAxis.labels.fontSize=10; plot.yValueAxis.labels.fontSize=10
    for i,color in enumerate(PALETTE): plot.lines[i].strokeColor=color; plot.lines[i].strokeWidth=2
    d.add(plot)
    for i,(m,c,label) in enumerate(series):
        color=PALETTE[i]
        for b in (1,2,4,8):
            lo,hi=group(m,c,b)["wilson95"]
            x=65+650*b/8; yl=86+230*lo; yh=86+230*hi
            d.add(Line(x,yl,x,yh,strokeColor=color,strokeWidth=1))
        lx=35+(i%2)*365; ly=44-(i//2)*19
        d.add(Line(lx,ly,lx+22,ly,strokeColor=color,strokeWidth=2))
        d.add(String(lx+28,ly-3,label,fontSize=10))
    d.add(String(270,64,"Additional phase-encoding columns",fontSize=11))
    d.add(String(12,328,"Flagged (%)",fontSize=10))
    d.add(String(18,5,"n=1536 test phantoms per point; Wilson intervals conditional on shared fitted model/calibration.",fontSize=8))
    chart_png(d,"audit_detection")


def market_figure():
    repos=json.loads((ROOT/"research/github_landscape.json").read_text(encoding="utf-8-sig"))["repositories"]
    repos=sorted(repos,key=lambda x:x["stars"],reverse=True)
    d=Drawing(760,345); d.add(Rect(0,0,760,345,fillColor=colors.white,strokeColor=None))
    d.add(String(18,320,"Open medical-imaging ecosystem: attention snapshot",fontName="Helvetica-Bold",fontSize=16))
    for i,r in enumerate(repos):
        y=270-i*39; w=430*r["stars"]/10000
        d.add(String(16,y+5,r["repository"],fontSize=9))
        d.add(Rect(270,y,w,18,fillColor=PALETTE[0],strokeColor=None))
        d.add(String(275+w,y+4,f"{r['stars']:,}",fontSize=10))
    d.add(String(18,18,"Stars at 3 Nov 2026 IST; purposive sample. Attention is not active adoption or a growth forecast.",fontSize=9))
    chart_png(d,"github_landscape")


def example_figure():
    with np.load(ROOT/"results/main/example.npz",allow_pickle=False) as e:
        arrays=[abs(e["truth"]),abs(e["oracle_insertion"]),e["template"].real,abs(e["cnn_insertion"])]
    image=PILImage.new("RGB",(1400,430),"white"); draw=ImageDraw.Draw(image)
    fontpath=Path("C:/Windows/Fonts/arial.ttf")
    font=ImageFont.truetype(str(fontpath),23) if fontpath.exists() else ImageFont.load_default()
    small=ImageFont.truetype(str(fontpath),18) if fontpath.exists() else ImageFont.load_default()
    names=["True magnitude","Oracle + insertion","Inserted error (real)","CNN + insertion"]
    for i,(a,label) in enumerate(zip(arrays,names)):
        if i==2:
            scale=max(abs(a).max(),1e-12); t=np.clip(a/scale,-1,1)
            rgb=np.stack([255*np.where(t>0,1,1+t),255*(1-abs(t)),255*np.where(t<0,1,1-t)],axis=-1)
            caption=f"Signed scale: +/- {scale:.2f}"
        else:
            t=np.clip(a,0,1); rgb=np.repeat((t*255)[...,None],3,axis=-1); caption="Magnitude scale: 0 to 1"
        panel=PILImage.fromarray(rgb.astype(np.uint8)).resize((320,320),PILImage.Resampling.NEAREST)
        image.paste(panel,(15+i*350,47)); draw.text((15+i*350,12),label,fill="#152935",font=font)
        draw.text((15+i*350,378),caption,fill="#374151",font=small)
    draw.text((15,408),"Synthetic example: fixed first test identity, seed 11. No patient data.",fill="#374151",font=small)
    image.save(ASSETS/"counterexample.png")


def manuscript():
    text=(ROOT/"paper/manuscript.template.md").read_text()
    cnn=group("cnn","insertion"); oracle=group("oracle","insertion")
    table=["| Reconstruction | Clean | Insertion | Removal | Single-line | Audit-aware |",
           "|---|---|---|---|---|---|"]
    for name in ("oracle","zero_fill","cnn"):
        cells=[]
        for case in ("clean","insertion","removal","single_line","audit_aware"):
            g=group(name,case); lo,hi=g["wilson95"]
            cells.append(f"{g['flagged']}/1536; {rate(g)} ({100*lo:.1f}-{100*hi:.1f})")
        table.append("| "+name+" | "+" | ".join(cells)+" |")
    policy=["| Reconstruction / case | Candidate | Fuse all | Selective fusion |",
            "|---|---:|---:|---:|"]
    for name in ("oracle","zero_fill","cnn"):
        for case in ("clean","insertion"):
            g=group(name,case)
            policy.append(f"| {name} / {case} | {g['mean_candidate_nrmse']:.4f} | {g['mean_fuse_all_nrmse']:.4f} | {g['mean_selective_nrmse']:.4f} |")
    values={"CNN_INSERT_COUNT":count(cnn),"CNN_INSERT_RATE":rate(cnn),
            "CNN_CLEAN_COUNT":count(group("cnn","clean")),"CNN_CLEAN_RATE":rate(group("cnn","clean")),
            "ORACLE_INSERT_COUNT":count(oracle),"ORACLE_INSERT_RATE":rate(oracle),
            "ORACLE_CLEAN_RATE":rate(group("oracle","clean")),"CNN_NATIVE":f"{cnn['mean_native_nrmse']:.4f}",
            "CNN_FUSE":f"{cnn['mean_fuse_all_nrmse']:.4f}","CNN_SELECTIVE":f"{cnn['mean_selective_nrmse']:.4f}",
            "CNN_RESIDUAL":f"{cnn['max_observed_residual']:.2e}","ORACLE_RESIDUAL":f"{oracle['max_observed_residual']:.2e}",
            "CNN_AUC":f"{cnn['score_auc_vs_clean']:.3f}","ORACLE_LINE_RATE":rate(group("oracle","single_line")),
            "ORACLE_NOISE_RATE":rate(group("oracle","noise_shift")),"CNN_NOISE_RATE":rate(group("cnn","noise_shift")),
            "CNN_ANATOMY_RATE":rate(group("cnn","anatomy_shift")),"CNN_POLICY_DELTA":f"{cnn['selective_minus_fuse_all']:.4f}",
            "CNN_POLICY_CI":" to ".join(f"{x:.4f}" for x in cnn["paired_bootstrap95"]),
            "PRIMARY_TABLE":"\n".join(table),"POLICY_TABLE":"\n".join(policy),"RUNTIME":f"{SUMMARY['elapsed_seconds']:.1f}"}
    for key,value in values.items(): text=text.replace("{{"+key+"}}",value)
    assert "{{" not in text
    (ROOT/"paper/manuscript.md").write_text(text,encoding="utf8")
    return text


def inline(text):
    text=html.escape(text)
    text=re.sub(r"\[([^\]]+)\]\(([^)]+)\)",lambda m:f'<a href="{m[2]}" color="#087f8c">{m[1]}</a>',text)
    text=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",text)
    text=re.sub(r"`([^`]+)`",r'<font name="Courier">\1</font>',text)
    return text


def make_pdf(markdown,path,title):
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyX",fontName="Helvetica",fontSize=9.7,leading=14.4,spaceAfter=7,alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="CellX",fontName="Helvetica",fontSize=7.2,leading=10,spaceAfter=0))
    styles["Title"].fontSize=23; styles["Title"].leading=28; styles["Title"].textColor=PALETTE[0]
    styles["Heading1"].fontSize=14; styles["Heading1"].leading=18; styles["Heading1"].spaceBefore=14
    styles["Heading2"].fontSize=11; styles["Heading2"].leading=15
    story=[]; lines=markdown.splitlines(); i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1; continue
        if line.startswith("```"):
            block=[]; i+=1
            while i<len(lines) and not lines[i].startswith("```"):
                block.append(lines[i]); i+=1
            style=ParagraphStyle(name="MonoBlock",fontName="Courier",fontSize=8,leading=12,spaceAfter=10,backColor=colors.HexColor("#eef4f5"),borderPadding=6)
            story.append(Preformatted("\n".join(block),style)); i+=1; continue
        if line.startswith("!["):
            source=re.search(r"\]\((.*?)\)",line).group(1)
            imagepath=ROOT/"paper"/source
            with PILImage.open(imagepath) as im: w,h=im.size
            story.append(Image(str(imagepath),width=480,height=480*h/w)); i+=1; continue
        if line.startswith("|"):
            table=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                cells=[c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+",c) for c in cells): table.append(cells)
                i+=1
            countcols=len(table[0]); widths=[480/countcols]*countcols
            wrapped=[[Paragraph(inline(c),styles["CellX"]) for c in row] for row in table]
            tab=Table(wrapped,colWidths=widths,repeatRows=1,hAlign="LEFT")
            tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5f1f3")),
                                     ("VALIGN",(0,0),(-1,-1),"TOP"),("LINEBELOW",(0,0),(-1,0),.6,PALETTE[0]),
                                     ("BOTTOMPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),6),
                                     ("LINEBELOW",(0,1),(-1,-1),.25,colors.HexColor("#d8e1e5"))]))
            story.extend([tab,Spacer(1,8)]); continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]),styles["Title"])); i+=1; continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]),styles["Heading1"])); i+=1; continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]),styles["Heading2"])); i+=1; continue
        paragraph=[line]; i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(("#","|","![","```")):
            paragraph.append(lines[i].strip()); i+=1
        story.append(Paragraph(inline(" ".join(paragraph)),styles["BodyX"]))
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor("#d5e0e3")); canvas.line(57,43,538,43)
        canvas.setFont("Helvetica",8); canvas.setFillColor(colors.HexColor("#52636c"))
        canvas.drawString(57,30,"K-Sentinel | Vinsy Moyya | Unreviewed research | 3 Nov 2026")
        canvas.drawRightString(538,30,str(doc.page))
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=57,leftMargin=57,topMargin=48,bottomMargin=57,
                          title=title,author="Vinsy Moyya")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(path)
    for page in reader.pages:
        text=page.extract_text() or ""
        assert "\u25a0" not in text and "{{" not in text
    print(f"{path.name}: {len(reader.pages)} pages")


def main():
    detection_figure(); market_figure(); example_figure()
    paper=manuscript()
    make_pdf(paper,OUTPUT/"k-sentinel-preprint.pdf","K-Sentinel: Limits of Held-Out Measurement Audits")
    market=(ROOT/"research/MARKET_ANALYSIS.md").read_text()
    marker="## Users, buyers and adoption constraints"
    market=market.replace(marker,"![GitHub landscape](../assets/github_landscape.png)\n\n"+marker)
    # The source figure path is resolved relative to paper/ in both documents.
    make_pdf(market,OUTPUT/"medical-imaging-ai-market-analysis.pdf","Medical Imaging AI: Opportunity and Adoption Analysis")


if __name__=="__main__": main()
