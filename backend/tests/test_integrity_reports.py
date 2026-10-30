"""Follow-up regressions; fixtures use only temporary storage and SQLite."""
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch
import test_reliability as fixtures
from app.models.analysis import MediaAnalysis, DeepfakeAnalysis, ElaAnalysis, FftAnalysis
from app.services.report_service import ReportService
from app.services.integrity_service import IntegrityService
from app.schemas.evidence import EvidenceResponse, IntegrityResponse
from app.api.routes.evidence import get_evidence_details, verify_evidence_integrity
from app.models.evidence import AuditLog
from app.services.report_pdf_service import ReportPdfService
from fastapi import HTTPException


class ReportSelectionTests(fixtures.IsolatedCase):
    def report_data(self, evidence):
        _, data = fixtures.ReportReliabilityTests.render_report(self, evidence)
        return data

    def test_new_pending_preprocessing_keeps_completed_linked_runs(self):
        ev = self.evidence(); completed = self.preprocessing(ev, 0)
        df = DeepfakeAnalysis(evidence_id=ev.id, preprocessing_analysis_id=completed.id,
            status='COMPLETED', model_name='test', model_version='NOT_CONFIGURED', overall_label='NO_FACE_DETECTED')
        self.db.add(df); self.db.commit()
        pending = MediaAnalysis(evidence_id=ev.id, status='PROCESSING', integrity_verified=True)
        self.db.add(pending); self.db.commit()
        data = self.report_data(ev)
        self.assertEqual(data['preprocessing']['run_id'], completed.id)
        self.assertEqual(data['preprocessing']['latest_attempt']['run_id'], pending.id)
        self.assertEqual(data['deepfake']['run_id'], df.id)

    def test_new_failed_signals_do_not_replace_completed_signals(self):
        ev = self.evidence()
        expected = {}
        latest = {}
        for kind, cls in [('ela', ElaAnalysis), ('fft', FftAnalysis)]:
            good = cls(evidence_id=ev.id, status='COMPLETED'); self.db.add(good); self.db.commit()
            failed = cls(evidence_id=ev.id, status='FAILED', error_message='test failure')
            self.db.add(failed); self.db.commit()
            expected[kind] = good.id; latest[kind] = failed.id
        # These metadata-only fixtures deliberately omit visualizations; exercise
        # selection without depending on artifact rendering in this test.
        with patch('app.services.report_service.ReportPdfService.generate_pdf',
                   side_effect=lambda data, path: Path(path).write_bytes(b'%PDF-test')):
            data = self.report_data(ev)
        for kind in ('ela', 'fft'):
            self.assertEqual(data[kind]['run_id'], expected[kind])
            self.assertEqual(data[kind]['latest_attempt']['run_id'], latest[kind])

    def test_no_face_takes_precedence_over_unconfigured_version(self):
        ev = self.evidence(); pre = self.preprocessing(ev, 0)
        self.db.add(DeepfakeAnalysis(evidence_id=ev.id, preprocessing_analysis_id=pre.id,
            status='COMPLETED', model_name='test', model_version='NOT_CONFIGURED', overall_label='NO_FACE_DETECTED'))
        self.db.commit()
        data = self.report_data(ev)
        self.assertEqual(data['deepfake']['overall_result'], 'NO_FACE_DETECTED')
        self.assertEqual(data['deepfake']['model_status'], 'NOT_RUN')

    def test_latest_completed_deepfake_used_with_new_failure_disclosed(self):
        ev = self.evidence(); pre = self.preprocessing(ev, 0)
        good = DeepfakeAnalysis(evidence_id=ev.id, preprocessing_analysis_id=pre.id,
            status='COMPLETED', model_name='test', model_version='test', overall_label='NO_FACE_DETECTED')
        self.db.add(good); self.db.commit()
        failed = DeepfakeAnalysis(evidence_id=ev.id, preprocessing_analysis_id=pre.id,
            status='FAILED', model_name='test', model_version='test', overall_label='MODEL_ERROR')
        self.db.add(failed); self.db.commit()
        data = self.report_data(ev)
        self.assertEqual(data['deepfake']['run_id'], good.id)
        self.assertEqual(data['deepfake']['latest_attempt']['run_id'], failed.id)
        self.assertIn('FAILED', ReportPdfService.interpretation(data))

    def test_status_matrix_remains_distinct(self):
        ev = self.evidence(); pre = self.preprocessing(ev, 0)
        data = self.report_data(ev)
        self.assertEqual(data['deepfake']['status'], 'NOT_RUN')
        cases = [('MODEL_NOT_CONFIGURED', 'COMPLETED', 'MODEL_NOT_CONFIGURED', 'NOT_CONFIGURED'),
                 ('MODEL_ERROR', 'FAILED', 'MODEL_ERROR', 'test'),
                 ('NO_FACE_DETECTED', 'COMPLETED', 'NOT_RUN', 'NOT_CONFIGURED'),
                 ('REAL', 'COMPLETED', 'CONFIGURED', 'test'),
                 ('FAKE', 'COMPLETED', 'CONFIGURED', 'test')]
        for label, status, expected_model, version in cases:
            with self.subTest(label=label):
                row = DeepfakeAnalysis(evidence_id=ev.id, preprocessing_analysis_id=pre.id,
                    status=status, model_name='fixture-only', model_version=version, overall_label=label)
                self.db.add(row); self.db.commit()
                data = self.report_data(ev)
                self.assertEqual(data['deepfake']['model_status'], expected_model)
                self.assertEqual(data['deepfake']['overall_result'], label)
                self.assertEqual(data['deepfake']['run_id'], row.id)
                text = ReportPdfService.interpretation(data)
                if label in ('REAL', 'FAKE'):
                    self.assertIn('Deepfake model output: ' + label, text)
                else:
                    self.assertIn('No overall classification', text)
                self.db.delete(row); self.db.commit()

    def test_failure_without_completed_run_remains_visible(self):
        ev = self.evidence()
        run = ElaAnalysis(evidence_id=ev.id, status='FAILED', error_message='decode failure')
        self.db.add(run); self.db.commit()
        data = self.report_data(ev)
        self.assertEqual(data['ela']['run_id'], run.id)
        self.assertEqual(data['ela']['status'], 'FAILED')


class IntegrityTests(fixtures.IsolatedCase):
    def test_registered_hash_in_detail_and_fresh_verification_response(self):
        ev = self.evidence()
        detail = EvidenceResponse.model_validate(get_evidence_details(ev.id, self.db))
        result = IntegrityResponse.model_validate(verify_evidence_integrity(ev.id, self.db))
        self.assertEqual(detail.sha256_hash, ev.sha256_hash)
        self.assertEqual(result.original_sha256, detail.sha256_hash)
        self.assertEqual(result.current_sha256, detail.sha256_hash)
        self.assertEqual(result.status, 'VERIFIED')
        self.assertTrue(result.matches)
        self.assertIsNotNone(result.checked_at.tzinfo)

    def test_mismatch_does_not_overwrite_reference_hash(self):
        ev = self.evidence(); expected = ev.sha256_hash
        Path(ev.storage_path).write_bytes(b'changed for isolated integrity test')
        result = IntegrityService.verify(self.db, ev.id)
        self.assertEqual(result['status'], 'MISMATCH')
        self.assertFalse(result['matches'])
        self.assertNotEqual(result['current_sha256'], expected)
        self.db.refresh(ev)
        self.assertEqual(ev.sha256_hash, expected)
        self.assertEqual(self.db.query(AuditLog).one().action, 'INTEGRITY_CHECK_MISMATCH')

    def test_missing_original_is_distinct_from_missing_record(self):
        ev = self.evidence(); Path(ev.storage_path).unlink()
        result = IntegrityService.verify(self.db, ev.id)
        self.assertEqual(result['status'], 'MISSING')
        self.assertIsNone(result['current_sha256'])
        self.assertIsNone(result['matches'])
        with self.assertRaises(HTTPException) as caught:
            IntegrityService.verify(self.db, ev.id + 1000)
        self.assertEqual(caught.exception.status_code, 404)

    def test_read_error_is_not_a_success(self):
        ev = self.evidence()
        with patch('app.services.integrity_service.HashingService.calculate_sha256', side_effect=OSError('locked')):
            with self.assertRaises(HTTPException) as caught:
                IntegrityService.verify(self.db, ev.id)
        self.assertEqual(caught.exception.status_code, 500)
        self.assertEqual(self.db.query(AuditLog).count(), 0)


if __name__ == '__main__':
    unittest.main()
