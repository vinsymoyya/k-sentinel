"""Train the open residual CNN from downloaded shards; keep test data held out."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .core import ifft, nrmse, threshold
from .audit import audit_reconstruction
from .dataset import load_partition, observed_kspace
from .learned import ResidualCNN, channels
from .model_io import reconstruct


def fit(train_data, seed=11, epochs=8):
    if not isinstance(epochs,int) or epochs < 1:
        raise ValueError("Epochs must be a positive integer")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    inputs=[]
    for i in range(len(train_data["identities"])):
        measured, _ = observed_kspace(train_data,i)
        inputs.append(ifft(measured))
    x=channels(np.stack(inputs)); y=channels(train_data["truth"])
    model=ResidualCNN()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    generator=torch.Generator().manual_seed(seed)
    history=[]
    model.train()
    for epoch in range(epochs):
        total=0.
        for ids in torch.randperm(len(x),generator=generator).split(32):
            optimizer.zero_grad()
            loss=((model(x[ids])-y[ids])**2).mean()
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            loss.backward(); optimizer.step()
            total+=loss.item()*len(ids)
        history.append(total/len(x))
        print(f"seed={seed} epoch={epoch+1}/{epochs} training_mse={history[-1]:.7f}",flush=True)
    model.eval()
    return model, history


def train_dataset(data_dir, output, seed=11, epochs=8, limit=None):
    parts={split:load_partition(data_dir,split,seed,limit) for split in ("train","calibration","test")}
    ids={split:data["identities"].tolist() for split,data in parts.items()}
    if any(set(ids[a]) & set(ids[b]) for a,b in (("train","calibration"),("train","test"),("calibration","test"))):
        raise ValueError("Training, calibration and test identities overlap")
    model,history=fit(parts["train"],seed,epochs)
    # The fixed-epoch model is frozen before any calibration/test inference.
    calibration=[]
    for i in range(len(parts["calibration"]["identities"])):
        data=parts["calibration"]
        measured,mask=observed_kspace(data,i)
        image=reconstruct(model,measured,mask)
        cols=data["audit_order"][i,:4]
        residual=np.fft.fftshift(np.fft.fft2(image,norm="ortho"))[:,cols]-data["audit_values"][i,:,:4]
        calibration.append(float(np.mean(abs(residual)**2)/float(data["sigma"])**2))
    test=[]
    for i,identity in enumerate(parts["test"]["identities"]):
        data=parts["test"]; measured,mask=observed_kspace(data,i)
        image=reconstruct(model,measured,mask)
        audit=audit_reconstruction(image,data["audit_order"][i,:4],data["audit_values"][i,:,:4],
                                   calibration,float(data["sigma"]))
        test.append({"identity":int(identity),"cnn_nrmse":nrmse(image,data["truth"][i]),
                     "zero_fill_nrmse":nrmse(ifft(measured),data["truth"][i]),"audit_flagged":audit["flagged"]})
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parent
    config={"seed":seed,"epochs":epochs,"torch":str(torch.__version__),"numpy":np.__version__,
            "architecture":"ResidualCNN-2-16-16-16-2", "identities":ids,
            "data_manifest_sha256":hashlib.sha256((Path(data_dir)/"manifest.json").read_bytes()).hexdigest(),
            "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))}}
    checkpoint=out/f"cnn_seed{seed}.pt"
    torch.save({"state_dict":model.state_dict(),"config":config,"history":history},checkpoint)
    report={"config":config,"training_mse":history,"calibration_scores":calibration,
            "audit_threshold":threshold(calibration) if len(calibration)>=19 else None,
            "test":test,"test_count":len(test),
            "mean_cnn_nrmse":float(np.mean([r["cnn_nrmse"] for r in test])),
            "mean_zero_fill_nrmse":float(np.mean([r["zero_fill_nrmse"] for r in test])),
            "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "claim_scope":"Synthetic single-coil benchmark; no patient or clinical validation"}
    (out/f"training_seed{seed}.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,default=11)
    parser.add_argument("--epochs",type=int,default=8)
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    report=train_dataset(args.data,args.output,args.seed,args.epochs,args.limit)
    print(json.dumps({key:report[key] for key in ("test_count","mean_cnn_nrmse","mean_zero_fill_nrmse","checkpoint_sha256")},indent=2))


if __name__=="__main__": main()
