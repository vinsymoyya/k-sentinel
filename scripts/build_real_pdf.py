"""Render the verified real-MRI Markdown addendum with readable wide figures."""
import html
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, NextPageTemplate,
    PageBreak, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether)
from PIL import Image as PILImage
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'research/real-acquisition/REPORT.md'
DESTINATION=ROOT/'output/pdf/k-sentinel-real-acquisition-addendum.pdf'


def rich(text):
    text=html.escape(text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`(.+?)`',r'<font name="Courier" size="8">\1</font>',text)
    return text


def main():
    DESTINATION.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyTextReal',fontName='Helvetica',fontSize=9.2,leading=13.2,
        textColor=colors.HexColor('#253346'),spaceAfter=7,splitLongWords=True))
    styles.add(ParagraphStyle(name='TitleReal',fontName='Helvetica-Bold',fontSize=23,leading=27,
        textColor=colors.HexColor('#075f66'),spaceAfter=12))
    styles.add(ParagraphStyle(name='HeadingReal',fontName='Helvetica-Bold',fontSize=12,leading=16,
        textColor=colors.HexColor('#075f66'),spaceBefore=12,spaceAfter=7,keepWithNext=True))
    styles.add(ParagraphStyle(name='TableReal',fontName='Helvetica',fontSize=7.6,leading=10,splitLongWords=True))
    styles.add(ParagraphStyle(name='CaptionReal',fontName='Helvetica',fontSize=9,leading=12,
        textColor=colors.HexColor('#475569'),spaceAfter=7))
    portrait=A4
    wide=landscape(A4)
    margins=43
    def header_footer(canvas,doc):
        width,height=canvas._pagesize
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#bfd6d8'))
        canvas.line(margins,height-33,width-margins,height-33)
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor('#48616b'))
        canvas.drawString(margins,height-25,'K-Sentinel | Exploratory real-acquisition addendum | 2026-11-05')
        canvas.drawString(margins,23,'Unreviewed research. Source-reference agreement is not clinical validation.')
        canvas.drawRightString(width-margins,23,str(doc.page))
        canvas.restoreState()
    doc=BaseDocTemplate(str(DESTINATION),pagesize=portrait,title='K-Sentinel: Real acquisition variations and natural reference-error failures',
        author='Vinsy Moyya',leftMargin=margins,rightMargin=margins,topMargin=48,bottomMargin=39)
    frames=[]
    for name,size in [('portrait',portrait),('wide',wide)]:
        frame=Frame(margins,39,size[0]-2*margins,size[1]-87,id=name,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        doc.addPageTemplates(PageTemplate(id=name,frames=frame,onPage=header_footer,pagesize=size))
    flow=[]
    lines=SOURCE.read_text(encoding='utf-8').splitlines()
    index=0
    while index<len(lines):
        line=lines[index].strip()
        if not line:
            index+=1;continue
        if line.startswith('!['):
            index+=1;continue
        if line=='## Figures':
            # Figure pages follow the text, with a larger landscape frame.
            index+=1
            while index<len(lines) and not lines[index].startswith('## References'):
                index+=1
            continue
        if line.startswith('|'):
            rows=[]
            while index<len(lines) and lines[index].strip().startswith('|'):
                parts=[p.strip() for p in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',p) for p in parts):
                    rows.append(parts)
                index+=1
            width=portrait[0]-2*margins
            columns=len(rows[0])
            widths=([width*.26]+[width*.74/(columns-1)]*(columns-1)) if columns>2 else [width/columns]*columns
            cells=[[Paragraph(rich(cell),styles['TableReal']) for cell in row] for row in rows]
            table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4f0ef')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#508b8e')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f9')]),
                ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            flow.extend([table,Spacer(1,9)])
            continue
        if line.startswith('# '):
            flow.append(Paragraph(rich(line[2:]),styles['TitleReal']))
        elif line.startswith('## '):
            flow.append(Paragraph(rich(line[3:]),styles['HeadingReal']))
        elif line.startswith('### '):
            flow.append(Paragraph(rich(line[4:]),styles['HeadingReal']))
        else:
            flow.append(Paragraph(rich(line),styles['BodyTextReal']))
        index+=1
    figures=[
        ('Sampling and audit','sampling_and_audit','T2 repetition 1, twenty-five held-out participants. Nominal acceleration is defined relative to actually acquired source lines. Wilson intervals condition on the fitted model and calibration.'),
        ('Actual acquisition variations','actual_acquisition_variations','Actual repeat acquisitions and contrast changes are separate from retrospective masks. Motion-risk groups are participant-level assignments, not file-level artifact truth labels. Every comparison shown uses the fine-tuned CNN and random R=4.'),
        ('Audit versus natural reference error','audit_vs_natural_error','Each point is one held-out participant, averaging three fixed slices. Vertical high-error cutoff and horizontal audit threshold are fit on thirty other calibration people. A small residual is not an anatomical certificate.'),
        ('Fixed-rule native failure cases','natural_failure_cases','Three highest-reference-error unflagged primary cases, selected by the stated fixed rule. Stored slice 8 is displayed; ranking averages slices 7/8/9. These primary cases did not exceed the high-error cutoff. Reference windowing is for display only. No inserted structures or biological lesion labels. Source MRI: M4Raw, Lyu et al. (2023), CC BY 4.0.'),
        ('Exploratory contrast failure cases','flair_unflagged_high_error','Both high-reference-error, unflagged cases in the preplanned FLAIR / random R=4 / repetition 1 secondary stratum. The illustrative selection was made after outcome inspection; it does not change primary outcomes, model or thresholds. Stored slice 8; ranking/labels use the mean of 7/8/9. Source MRI: M4Raw, Lyu et al. (2023), CC BY 4.0. No diagnostic or focal-lesion truth labels.')]
    flow.append(NextPageTemplate('wide'))
    for number,(title,name,caption) in enumerate(figures,1):
        flow.append(PageBreak())
        flow.append(Paragraph(f'Figure {number}. {title}',styles['HeadingReal']))
        path=ROOT/f'assets/real-acquisition/{name}.png'
        with PILImage.open(path) as image:
            width,height=image.size
        maximum_width=wide[0]-2*margins
        maximum_height=wide[1]-188
        scale=min(maximum_width/width,maximum_height/height)
        flow.append(Image(str(path),width=width*scale,height=height*scale,hAlign='CENTER'))
        flow.append(Spacer(1,7))
        flow.append(Paragraph(rich(caption),styles['CaptionReal']))
    doc.build(flow)
    reader=PdfReader(DESTINATION)
    text='\n'.join(page.extract_text() or '' for page in reader.pages)
    for term in ['180','100','Source','calibration','M4Raw']:
        assert term in text,term
    print(f'{DESTINATION} | {len(reader.pages)} pages')


if __name__=='__main__':
    main()
