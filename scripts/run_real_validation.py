"""Retain per-case evidence for the additional real-acquisition contracts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='validation/local/real')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests_real'))
    names = [test.id() for test in flatten(suite)]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failures = {test.id(): detail for test, detail in result.failures + result.errors}
    skips = {test.id(): reason for test, reason in result.skipped}
    passed = result.testsRun - len(failures) - len(skips)
    records = [{'test': name, 'status': 'failed' if name in failures else 'skipped' if name in skips
                else 'passed' if result.testsRun == len(names) else 'unverified',
                **({'detail': failures[name]} if name in failures else {})} for name in names]
    report = {'total': len(names), 'executed': result.testsRun, 'passed': max(0, passed),
        'failed': len(names) - max(0, passed) - len(skips), 'skipped': len(skips),
        'status': 'passed' if names and passed == len(names) and result.wasSuccessful() else 'failed',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': records,
        'scope': 'Additional real-acquisition software contracts; clinical validation not performed'}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    xml = ET.Element('testsuite', name='K-Sentinel real-acquisition contracts', tests=str(len(names)),
                     failures=str(report['failed']), skipped=str(report['skipped']))
    for record in records:
        case = ET.SubElement(xml, 'testcase', name=record['test'])
        if record['status'] != 'passed':
            ET.SubElement(case, 'skipped' if record['status']=='skipped' else 'failure').text = record.get('detail', record['status'])
    ET.ElementTree(xml).write(output/'junit.xml', encoding='utf-8', xml_declaration=True)
    print(json.dumps({key:report[key] for key in ('total','passed','failed','skipped','status')}, indent=2))
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    sys.exit(main())
