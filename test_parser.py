import json
import unittest
from pathlib import Path

from parser import parse_query


SCHEMA_PATH = Path(__file__).resolve().parent / "full_schema.json"
with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
    SCHEMA = json.load(handle)


class ParserPipelineTests(unittest.TestCase):

    def test_basic_filter(self) -> None:
        result = parse_query("show events where up_event_duration > 30")

        self.assertEqual(result["collection"], "cycle_events.events")
        self.assertEqual(result["operation"], "find")
        self.assertEqual(result["filter"], {"up_event_duration": {"$gt": 30.0}})

    def test_logical_and_sort_limit(self) -> None:
        result = parse_query(
            "show events where camera is CAM001 and duration greater than 10 "
            "sort by up_event_start desc limit 5"
        )

        self.assertEqual(result["collection"], "cycle_events.events")
        self.assertEqual(result["operation"], "find")
        self.assertEqual(
            result["filter"],
            {
                "$and": [
                    {"camera_id": "CAM001"},
                    {"up_event_duration": {"$gt": 10.0}},
                ]
            },
        )
        self.assertEqual(result["sort"], {"up_event_start": -1})
        self.assertEqual(result["limit"], 5)

    def test_logical_or(self) -> None:
        result = parse_query("show events where camera is CAM001 or camera is CAM002")

        self.assertEqual(
            result["filter"],
            {"$or": [{"camera_id": "CAM001"}, {"camera_id": "CAM002"}]},
        )

    def test_count_intent(self) -> None:
        result = parse_query("count events where camera is CAM001")

        self.assertEqual(result["collection"], "cycle_events.events")
        self.assertEqual(result["operation"], "countDocuments")
        self.assertEqual(result["filter"], {"camera_id": "CAM001"})

    def test_backward_compatible_schema_arg(self) -> None:
        result = parse_query("show events where camera is CAM001", schema=SCHEMA)

        self.assertEqual(result["collection"], "cycle_events.events")
        self.assertEqual(result["filter"], {"camera_id": "CAM001"})

    def test_empty_query_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_query("   ")


if __name__ == "__main__":
    unittest.main()

