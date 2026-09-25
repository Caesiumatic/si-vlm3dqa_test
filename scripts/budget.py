"""Leave a final reporting buffer within the user's wall-clock deadline."""
import datetime
import json
import pathlib


def check_deadline():
    protocol=json.loads(pathlib.Path('configs/protocol.json').read_text())
    deadline=datetime.datetime.fromisoformat(protocol['deadline_utc'].replace('Z','+00:00'))
    if (deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()<1800:
        raise SystemExit('DEADLINE_REPORTING_BUFFER: no new model work')
