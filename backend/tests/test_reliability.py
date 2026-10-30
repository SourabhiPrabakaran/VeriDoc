"""Isolated regression checks. Never use the workstation's evidence database."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.database import Base
from app.models.evidence import Evidence, AuditLog
from app.models.analysis import MediaAnalysis, FaceDetection, DeepfakeAnalysis, DeepfakePrediction, ElaAnalysis, FftAnalysis
from app.models.report import ForensicReport
from app.services.deepfake_model_interface import DeepfakeModelInterface, MesoInception4
from app.services.deepfake_detection_service import DeepfakeDetectionService
from app.services.report_service import ReportService
from app.services.report_pdf_service import ReportPdfService
from app.services.ela_service import ElaService
from fastapi import HTTPException
import json


class IsolatedCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.saved = {k: getattr(settings, k) for k in
                      ('BACKEND_DIR', 'DEEPFAKE_MODEL_PATH', 'DEEPFAKE_MODEL_VERSION', 'DEEPFAKE_DEVICE')}
        settings.BACKEND_DIR = self.root
        settings.DEEPFAKE_DEVICE = 'cpu'
        self.engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()
        DeepfakeModelInterface._model = None
        DeepfakeModelInterface._model_status = 'MODEL_NOT_CONFIGURED'
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.db.close()
        self.engine.dispose()
        for key, value in self.saved.items():
            setattr(settings, key, value)
        DeepfakeModelInterface._model = None
        DeepfakeModelInterface._model_status = 'MODEL_NOT_CONFIGURED'
        self.temp.cleanup()

    def evidence(self, gray=False):
        image = np.full((48, 48) if gray else (48, 48, 3), 120, dtype=np.uint8)
        path = self.root / 'original.png'
        self.assertTrue(cv2.imwrite(str(path), image))
        ev = Evidence(original_filename='original.png', stored_filename='original.png',
                      media_category='IMAGE', mime_type='image/png', file_size=path.stat().st_size,
                      file_extension='.png', sha256_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
                      storage_path=str(path), status='REGISTERED')
        self.db.add(ev); self.db.commit()
        return ev

    def preprocessing(self, ev, count=1):
        pre = MediaAnalysis(evidence_id=ev.id, status='COMPLETED', integrity_verified=True,
                            analysis_type='PREPROCESSING_FACE_DETECTION')
        self.db.add(pre); self.db.flush()
        for i in range(count):
            self.db.add(FaceDetection(analysis_id=pre.id, face_index=i, x=0, y=0,
                                     width=32, height=32, crop_path=f'missing_{i}.jpg'))
        self.db.commit()
        return pre

    def checkpoint(self, state=None, name='model.pth'):
        path = self.root / name
        torch.save(MesoInception4().state_dict() if state is None else state, path)
        settings.DEEPFAKE_MODEL_PATH = str(path)
        settings.DEEPFAKE_MODEL_VERSION = 'test-only-random-weights'
        DeepfakeModelInterface._model = None
        return path


class InferenceReliabilityTests(IsolatedCase):
    def test_missing_crop_with_loaded_model_never_becomes_real(self):
        ev = self.evidence(); self.preprocessing(ev); self.checkpoint()
        run = DeepfakeDetectionService.run_deepfake_analysis(self.db, ev.id)
        self.assertEqual(run.predictions[0].label, 'MODEL_ERROR')
        self.assertEqual(run.overall_label, 'MODEL_ERROR')
        self.assertEqual(run.status, 'FAILED')
        self.assertIsNone(run.overall_confidence)

    def test_partial_failure_never_becomes_real_or_fake(self):
        ev = self.evidence(); self.preprocessing(ev, 2); self.checkpoint()
        for valid_label in ('REAL', 'FAKE'):
            with self.subTest(valid_label=valid_label):
                predictions = [dict(label=valid_label, confidence=.8, model_status='CONFIGURED'),
                               dict(label='MODEL_ERROR', confidence=None, model_status='MODEL_ERROR')]
                with patch.object(DeepfakeModelInterface, 'predict', side_effect=predictions):
                    run = DeepfakeDetectionService.run_deepfake_analysis(self.db, ev.id)
                self.assertEqual(run.overall_label, 'MODEL_ERROR')
                self.assertEqual(run.status, 'FAILED')
                self.assertEqual(len(run.predictions), 2)
                self.assertIsNone(run.overall_confidence)

    def test_successful_classifications_preserved(self):
        ev = self.evidence(); self.preprocessing(ev, 2); self.checkpoint()
        for labels, expected in [(('REAL', 'REAL'), 'REAL'), (('REAL', 'FAKE'), 'FAKE')]:
            with self.subTest(labels=labels), patch.object(DeepfakeModelInterface, 'predict', side_effect=[
                dict(label=l, confidence=.8, model_status='CONFIGURED') for l in labels]):
                run = DeepfakeDetectionService.run_deepfake_analysis(self.db, ev.id)
                self.assertEqual(run.overall_label, expected)
                self.assertEqual(run.status, 'COMPLETED')

    def test_invalid_prediction_confidence_is_failure(self):
        ev = self.evidence(); self.preprocessing(ev); self.checkpoint()
        for confidence in (None, float('nan'), float('inf'), -1, 2):
            with self.subTest(confidence=confidence), patch.object(DeepfakeModelInterface, 'predict',
                    return_value=dict(label='REAL', confidence=confidence, model_status='CONFIGURED')):
                run = DeepfakeDetectionService.run_deepfake_analysis(self.db, ev.id)
                self.assertEqual(run.overall_label, 'MODEL_ERROR')

    def test_no_face_remains_no_face(self):
        ev = self.evidence(); self.preprocessing(ev, 0)
        run = DeepfakeDetectionService.run_deepfake_analysis(self.db, ev.id)
        self.assertEqual(run.overall_label, 'NO_FACE_DETECTED')

    def test_raised_inference_exception_sets_failed_error_state(self):
        ev = self.evidence(); self.preprocessing(ev); self.checkpoint()
        with patch.object(DeepfakeModelInterface, 'predict', side_effect=RuntimeError('device failed')):
            with self.assertRaises(HTTPException):
                DeepfakeDetectionService.run_deepfake_analysis(self.db, ev.id)
        run = self.db.query(DeepfakeAnalysis).one()
        self.assertEqual((run.status, run.overall_label), ('FAILED', 'MODEL_ERROR'))

    def test_disabled_model_cannot_reuse_previous_status(self):
        ev = self.evidence(); self.preprocessing(ev)
        settings.DEEPFAKE_MODEL_PATH = None
        settings.DEEPFAKE_MODEL_VERSION = 'NOT_CONFIGURED'
        DeepfakeModelInterface._model_status = 'CONFIGURED'
        run = DeepfakeDetectionService.run_deepfake_analysis(self.db, ev.id)
        self.assertEqual(run.overall_label, 'MODEL_NOT_CONFIGURED')
        self.assertEqual(run.predictions[0].label, 'MODEL_NOT_CONFIGURED')


class CheckpointTests(IsolatedCase):
    def assert_rejected(self, state):
        self.checkpoint(state)
        self.assertIsNone(DeepfakeModelInterface.load_model())
        self.assertEqual(DeepfakeModelInterface._model_status, 'MODEL_ERROR')

    def test_partial_checkpoint_rejected(self):
        self.assert_rejected({'bn1.bias': torch.zeros(11)})

    def test_wrong_shape_rejected(self):
        state = MesoInception4().state_dict(); state['bn1.bias'] = torch.zeros(12)
        self.assert_rejected(state)

    def test_unexpected_keys_rejected(self):
        state = MesoInception4().state_dict(); state['unexpected'] = torch.zeros(1)
        self.assert_rejected(state)

    def test_nonfinite_checkpoint_rejected(self):
        state = MesoInception4().state_dict(); state['bn1.bias'][0] = float('nan')
        self.assert_rejected(state)

    def test_complete_plain_and_wrapped_checkpoint(self):
        for wrapper in (None, 'state_dict', 'model'):
            with self.subTest(wrapper=wrapper):
                state = MesoInception4().state_dict()
                self.checkpoint(state if wrapper is None else {wrapper: state})
                self.assertIsNotNone(DeepfakeModelInterface.load_model())

    def test_settings_change_invalidates_cached_model(self):
        self.checkpoint(); self.assertIsNotNone(DeepfakeModelInterface.load_model())
        settings.DEEPFAKE_MODEL_PATH = str(self.root / 'missing.pth')
        self.assertIsNone(DeepfakeModelInterface.load_model())

    def test_wrong_dtype_rejected(self):
        state = MesoInception4().state_dict(); state['bn1.bias'] = torch.zeros(11, dtype=torch.float64)
        self.assert_rejected(state)

    def test_overwrite_same_path_invalidates_cache(self):
        path = self.checkpoint(); self.assertIsNotNone(DeepfakeModelInterface.load_model())
        torch.save({'bn1.bias': torch.zeros(11)}, path)
        self.assertIsNone(DeepfakeModelInterface.load_model())


class ReportReliabilityTests(IsolatedCase):
    def render_report(self, evidence):
        original = ReportPdfService.generate_pdf
        captured = {}
        def capture(data, path):
            captured.update(data)
            return original(data, path)
        with patch.object(ReportPdfService, 'generate_pdf', side_effect=capture):
            report = ReportService.generate_report(self.db, evidence.id)
        return report, captured

    def test_tampered_original_rejected_even_after_verified_preprocessing(self):
        ev = self.evidence(); self.preprocessing(ev, 0)
        Path(ev.storage_path).write_bytes(b'changed after preprocessing')
        with self.assertRaises(HTTPException) as caught:
            ReportService.generate_report(self.db, ev.id)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(self.db.query(ForensicReport).filter_by(status='COMPLETED').count(), 0)

    def test_missing_original_rejected(self):
        ev = self.evidence(); Path(ev.storage_path).unlink()
        with self.assertRaises(HTTPException) as caught:
            ReportService.generate_report(self.db, ev.id)
        self.assertEqual(caught.exception.status_code, 404)

    def test_report_with_no_analyses_has_fresh_integrity_and_not_run_states(self):
        report, data = self.render_report(self.evidence())
        self.assertEqual(report.status, 'COMPLETED')
        self.assertEqual(data['evidence_integrity_status'], 'VERIFIED')
        for name in ('preprocessing', 'deepfake', 'ela', 'fft'):
            self.assertEqual(data[name]['status'], 'NOT_RUN')
        self.assertNotIn('successfully underwent', ReportPdfService.interpretation(data))
        self.assertIn('NOT_RUN', ReportPdfService.interpretation(data))
        snapshot = json.loads(Path(report.file_path).with_suffix('.json').read_text())
        self.assertEqual(snapshot['evidence']['sha256_hash'], data['evidence']['sha256_hash'])
        self.assertEqual(hashlib.sha256(Path(report.file_path).read_bytes()).hexdigest(), report.file_hash)

    def test_new_preprocessing_does_not_mix_old_deepfake_run(self):
        ev = self.evidence(); old = self.preprocessing(ev, 0)
        self.db.add(DeepfakeAnalysis(evidence_id=ev.id, preprocessing_analysis_id=old.id,
            status='COMPLETED', model_name='test', model_version='test', overall_label='REAL'))
        self.db.commit()
        new = self.preprocessing(ev, 0)
        report, data = self.render_report(ev)
        self.assertEqual(data['preprocessing']['run_id'], new.id)
        self.assertEqual(data['deepfake']['status'], 'NOT_RUN')
        self.assertIsNone(data['deepfake']['run_id'])

    def test_model_error_report_is_not_configured_or_classified(self):
        ev = self.evidence(); pre = self.preprocessing(ev, 0)
        df = DeepfakeAnalysis(evidence_id=ev.id, preprocessing_analysis_id=pre.id,
            status='FAILED', model_name='test', model_version='test', overall_label='MODEL_ERROR',
            error_message='crop missing')
        self.db.add(df); self.db.commit()
        _, data = self.render_report(ev)
        self.assertEqual(data['deepfake']['run_id'], df.id)
        self.assertEqual(data['deepfake']['model_status'], 'MODEL_ERROR')
        self.assertIn('No overall classification', ReportPdfService.interpretation(data))

    def test_legacy_false_real_row_with_error_prediction_is_not_repeated(self):
        ev = self.evidence(); pre = self.preprocessing(ev)
        df = DeepfakeAnalysis(evidence_id=ev.id, preprocessing_analysis_id=pre.id,
            status='COMPLETED', model_name='test', model_version='test', overall_label='REAL')
        self.db.add(df); self.db.flush()
        self.db.add(DeepfakePrediction(analysis_id=df.id, face_detection_id=pre.faces[0].id,
            label='MODEL_ERROR', model_status='MODEL_ERROR', confidence=None))
        self.db.commit()
        _, data = self.render_report(ev)
        self.assertEqual(data['deepfake']['overall_result'], 'MODEL_ERROR')
        self.assertIn('No overall classification', ReportPdfService.interpretation(data))

    def test_success_summary_and_run_ids_are_actual(self):
        ev = self.evidence(); pre = self.preprocessing(ev, 0)
        df = DeepfakeAnalysis(evidence_id=ev.id, preprocessing_analysis_id=pre.id,
            status='COMPLETED', model_name='test', model_version='test', overall_label='FAKE',
            overall_confidence=.75)
        self.db.add(df); self.db.commit()
        _, data = self.render_report(ev)
        self.assertEqual(data['deepfake']['run_id'], df.id)
        text = ReportPdfService.interpretation(data)
        self.assertIn('FAKE', text)
        self.assertNotIn('weights are unavailable', text)
        self.assertIn('ELA: NOT_RUN', text)

    def test_tamper_during_render_fails_without_completion_event(self):
        ev = self.evidence()
        original = ReportPdfService.generate_pdf
        def tamper(data, path):
            original(data, path)
            Path(ev.storage_path).write_bytes(b'changed while rendering')
        with patch.object(ReportPdfService, 'generate_pdf', side_effect=tamper):
            with self.assertRaises(HTTPException) as caught:
                ReportService.generate_report(self.db, ev.id)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(self.db.query(AuditLog).filter_by(action='REPORT_GENERATION_COMPLETED').count(), 0)
        report = self.db.query(ForensicReport).one()
        self.assertEqual(report.status, 'FAILED')
        self.assertFalse(Path(report.file_path).exists())
        self.assertFalse(Path(report.file_path).with_suffix('.json').exists())

    def test_unsafe_filename_markup_is_literal(self):
        ev = self.evidence(); ev.original_filename = 'name <not-a-tag> & test.png'; self.db.commit()
        report, _ = self.render_report(ev)
        self.assertEqual(report.status, 'COMPLETED')

    def test_render_failure_removes_outputs_and_does_not_log_completion(self):
        ev = self.evidence()
        with patch.object(ReportPdfService, 'generate_pdf', side_effect=RuntimeError('render failed')):
            with self.assertRaises(HTTPException):
                ReportService.generate_report(self.db, ev.id)
        report = self.db.query(ForensicReport).one()
        self.assertEqual(report.status, 'FAILED')
        self.assertFalse(Path(report.file_path).with_suffix('.json').exists())
        self.assertEqual(self.db.query(AuditLog).filter_by(action='REPORT_GENERATION_COMPLETED').count(), 0)


class ElaReliabilityTests(IsolatedCase):
    def test_grayscale_ela_completes_and_preserves_original(self):
        ev = self.evidence(gray=True)
        run = ElaService.run_ela(self.db, ev.id)
        self.assertEqual(run.status, 'COMPLETED')
        self.assertTrue((settings.ELA_DIR / run.visualization_path).exists())
        self.assertEqual(hashlib.sha256(Path(ev.storage_path).read_bytes()).hexdigest(), ev.sha256_hash)
        self.assertEqual(list(settings.TEMP_DIR.iterdir()), [])

    def test_16bit_input_fails_explicitly(self):
        ev = self.evidence()
        cv2.imwrite(ev.storage_path, np.full((24, 24), 40000, dtype=np.uint16))
        ev.sha256_hash = hashlib.sha256(Path(ev.storage_path).read_bytes()).hexdigest(); self.db.commit()
        with self.assertRaises(HTTPException) as caught:
            ElaService.run_ela(self.db, ev.id)
        self.assertIn('8-bit', caught.exception.detail)
        self.assertEqual(self.db.query(ElaAnalysis).one().status, 'FAILED')

    def test_failed_image_write_is_not_completed_and_temp_is_cleaned(self):
        ev = self.evidence()
        with patch('cv2.imwrite', return_value=False), self.assertRaises(HTTPException):
            ElaService.run_ela(self.db, ev.id)
        self.assertEqual(self.db.query(ElaAnalysis).one().status, 'FAILED')
        self.assertEqual(list(settings.TEMP_DIR.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
