import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CoachAppStaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "coach_app/static/index.html").read_text(encoding="utf-8")
        cls.javascript = (ROOT / "coach_app/static/app.js").read_text(encoding="utf-8")
        cls.styles = (ROOT / "coach_app/static/styles.css").read_text(encoding="utf-8")

    def test_session_participant_controls_have_unique_serbian_latin_contracts(self):
        for control_id in (
            "trainer-name",
            "wrestler-name",
            "save-participants-button",
        ):
            self.assertEqual(self.html.count(f'id="{control_id}"'), 1)
        for label in (
            "Ime trenera",
            "Ime rvača",
            "Sačuvaj podatke",
            "Preuzmi skup (JSON)",
            "Preuzmi audit (JSON)",
        ):
            self.assertIn(label, self.html)
        self.assertEqual(self.html.count('maxlength="120"'), 3)

    def test_session_participant_controls_link_clean_and_audit_json_exports(self):
        self.assertIn('href="/trener_dataset.json"', self.html)
        self.assertIn('href="/trener_assessment_audit.json"', self.html)

    def test_client_saves_participants_through_the_session_endpoint(self):
        self.assertIn("/api/session/participants", self.javascript)
        self.assertIn('method: "PUT"', self.javascript)
        self.assertIn("trainer_name: trainerName", self.javascript)
        self.assertIn("wrestler_name: wrestlerName", self.javascript)

    def test_session_identity_layout_collapses_to_one_column_on_narrow_screens(self):
        self.assertIn(".session-identity-grid", self.styles)
        self.assertIn("@media (max-width: 620px)", self.styles)


if __name__ == "__main__":
    unittest.main()
