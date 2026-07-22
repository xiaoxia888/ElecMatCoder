import unittest

from src.llm_ner.stage1_orchestrator import Stage1FieldOrchestrator


class _Predictor:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        return self.output


class _Router:
    def route(self, _text):
        return {
            "category": "管件",
            "confidence": 0.99,
            "source": "test_router",
            "candidates": [],
        }


def _field_output(field, value, *, category=""):
    model_output = {field: value}
    if category:
        model_output["CATEGORY"] = category
    return {
        "model_output": model_output,
        "model_raw_response": "",
        "extract_confidence": {field: 1.0},
        "extract_confidence_v2": {},
    }


class ImportedCategoryTest(unittest.TestCase):
    def setUp(self):
        self.pipe_type = _Predictor(
            _field_output("TYPE", {"BODY": "直管", "MANU": []}, category="直管")
        )
        self.fitting_type = _Predictor(
            _field_output("TYPE", {"BODY": "弯头", "MANU": []}, category="管件")
        )
        self.material = _Predictor(_field_output("MATERIAL", []))
        self.standard = _Predictor(_field_output("STANDARD", []))
        self.orchestrator = Stage1FieldOrchestrator(
            router=_Router(),
            type_factories={
                "直管": lambda: self.pipe_type,
                "管件": lambda: self.fitting_type,
            },
            default_type_factory=lambda: self.fitting_type,
            material_factory=lambda: self.material,
            standard_factory=lambda: self.standard,
            encodable_categories={"直管", "管件", "法兰"},
        )

    def test_imported_category_overrides_router_and_selects_type_model(self):
        result = self.orchestrator.predict("PIPE DN100", category_override=" 直管 ")

        self.assertEqual(result["route_info"]["category"], "直管")
        self.assertEqual(result["route_info"]["source"], "imported_category")
        self.assertEqual(result["route_info"]["route_level"], "imported")
        self.assertEqual(result["route_info"]["model_category"], "直管")
        self.assertEqual(len(self.pipe_type.calls), 1)
        self.assertEqual(len(self.fitting_type.calls), 0)

    def test_missing_imported_category_falls_back_to_router(self):
        result = self.orchestrator.predict("ELBOW DN100")

        self.assertEqual(result["route_info"]["category"], "管件")
        self.assertEqual(len(self.fitting_type.calls), 1)
        self.assertEqual(len(self.pipe_type.calls), 0)


if __name__ == "__main__":
    unittest.main()
