import tempfile
import unittest
from pathlib import Path

from pipeline.video_review_reports import markdown_cell, write_reports


class VideoReviewReportTests(unittest.TestCase):
    def test_markdown_cell_normalizes_mixed_newlines_and_escapes_backslashes_first(self):
        rendered = markdown_cell("putanja\\snimak | prvi\r\ndrugi\rtreci\n<kraj>")

        self.assertEqual(
            rendered,
            "putanja\\\\snimak \\| prvi<br>drugi<br>treci<br>&lt;kraj&gt;",
        )

    def test_markdown_report_keeps_mixed_content_inside_one_table_row(self):
        review = {
            "events": [
                {
                    "event_id": "e-1",
                    "sony_start_s": 10.0,
                    "sony_end_s": 12.0,
                    "napomena": "linija 1\r\nlinija 2 | C:\\video\rlinija 3",
                    "iskljuceno_iz_statistike": False,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as raw:
            review_path = Path(raw) / "review.json"
            write_reports(review_path, review)
            markdown = review_path.with_name("izvestaj.md").read_text(encoding="utf-8")

        self.assertNotIn("\r", markdown)
        self.assertIn("linija 1<br>linija 2 \\| C:\\\\video<br>linija 3", markdown)
        self.assertEqual(sum(line.startswith("| e-1 |") for line in markdown.splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
