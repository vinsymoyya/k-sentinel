"""Run exactly 45 named validation cases; skips and missing cases do not pass."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
EXISTING={"test_unitarity_and_inverse":1,"test_projection_cannot_see_injected_structure":2,
          "test_frozen_audit_can_be_evaded":3,"test_audit_fraction_and_independence":4,
          "test_exact_rank_threshold_and_ties":5,"test_wilson_known_boundary_and_auc":6,
          "test_measured_fusion_reduces_noiseless_error":7,"test_analytic_signal_rejection":8,
          "test_no_information_does_not_imply_safe":9,"test_invalid_masks_and_values":10}


def check_id(test):
    name=test.id().split(".")[-1]
    match=re.search(r"test_ks(\d{3})_",name)
    return int(match[1]) if match else EXISTING.get(name)


def flatten(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from flatten(item)
        else:yield item


class Result(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.records={}
    def startTest(self,test):
        super().startTest(test)
        self.records[test.id()]={"id":f"KS{check_id(test):03d}","test":test.id(),"status":"running","started":time.perf_counter()}
    def stopTest(self,test):
        record=self.records[test.id()];record["seconds"]=time.perf_counter()-record.pop("started")
        super().stopTest(test)
    def addSuccess(self,test):super().addSuccess(test);self.records[test.id()]["status"]="passed"
    def addFailure(self,test,err):super().addFailure(test,err);self.records[test.id()].update(status="failed",detail=self._exc_info_to_string(err,test))
    def addError(self,test,err):
        super().addError(test,err)
        if test.id() in self.records:self.records[test.id()].update(status="failed",detail=self._exc_info_to_string(err,test))
    def addSkip(self,test,reason):super().addSkip(test,reason);self.records[test.id()].update(status="skipped",detail=reason)
    def addSubTest(self,test,subtest,err):
        super().addSubTest(test,subtest,err)
        if err is not None:self.records[test.id()].update(status="failed",detail=self._exc_info_to_string(err,test))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="validation/local")
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    tests=list(flatten(suite));ids=[check_id(test) for test in tests]
    if len(tests)!=45 or set(ids)!=set(range(1,46)):
        raise SystemExit(f"Expected exactly KS001-KS045, discovered {len(tests)} cases with ids {ids}")
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    records=sorted(result.records.values(),key=lambda r:r["id"])
    passed=sum(r["status"]=="passed" for r in records)
    skipped=sum(r["status"]=="skipped" for r in records)
    try:commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):commit=None
    import numpy,torch
    report={"total":45,"executed":result.testsRun,"passed":passed,"failed":45-passed-skipped,"skipped":skipped,
            "status":"passed" if passed==45 and result.wasSuccessful() else "failed",
            "recorded_at_utc":datetime.now(timezone.utc).isoformat(),"source_commit":commit,
            "environment":{"python":platform.python_version(),"numpy":numpy.__version__,"torch":str(torch.__version__),"platform":platform.system()},
            "source_sha256":{str(p.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ("ksentinel","tests","scripts") for p in sorted((ROOT/directory).glob("*.py"))},
            "checks":records,"scope":"Software and synthetic scientific contracts; not clinical validation or an estimate of medical reliability"}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    xml=ET.Element("testsuite",name="K-Sentinel KS001-KS045",tests="45",failures=str(report["failed"]),skipped=str(skipped))
    for record in records:
        child=ET.SubElement(xml,"testcase",name=record["id"]+" "+record["test"],time=str(record.get("seconds",0)))
        if record["status"]!="passed":ET.SubElement(child,"skipped" if record["status"]=="skipped" else "failure").text=record.get("detail","Case did not complete")
    ET.ElementTree(xml).write(output/"junit.xml",encoding="utf-8",xml_declaration=True)
    print(json.dumps({key:report[key] for key in ("total","executed","passed","failed","skipped","status")},indent=2))
    return 0 if report["status"]=="passed" else 1


if __name__=="__main__":sys.exit(main())
