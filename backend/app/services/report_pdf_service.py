import os
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from app.core.config import settings

class ReportPdfService:
    @staticmethod
    def generate_pdf(report_data: dict, output_path: str):
        """
        Generates a professional forensic PDF report utilizing ReportLab flowables.
        """
        # Page geometry setup: standard letter size with 0.75-inch margins
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        styles = getSampleStyleSheet()
        
        # Define clean forensic branding colors
        primary_color = colors.HexColor("#00f0ff") # Forensic cyan
        dark_bg = colors.HexColor("#0f172a") # Dark slate
        text_light = colors.HexColor("#f8fafc") # High contrast light grey
        text_dark = colors.HexColor("#334155")
        accent_color = colors.HexColor("#e2e8f0")
        red_tint = colors.HexColor("#ef4444")
        green_tint = colors.HexColor("#22c55e")
        amber_tint = colors.HexColor("#f59e0b")

        # Custom ParagraphStyles
        title_style = ParagraphStyle(
            'CoverTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=32,
            textColor=colors.HexColor("#1e293b"),
            leading=38,
            alignment=0, # Left aligned
            spaceAfter=15
        )

        subtitle_style = ParagraphStyle(
            'CoverSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=16,
            textColor=colors.HexColor("#475569"),
            leading=22,
            alignment=0,
            spaceAfter=40
        )

        h1_style = ParagraphStyle(
            'Heading1Custom',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=15,
            spaceAfter=10,
            keepWithNext=True
        )

        body_style = ParagraphStyle(
            'BodyTextCustom',
            parent=styles['BodyText'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor("#334155"),
            leading=14,
            spaceAfter=8
        )

        body_bold = ParagraphStyle(
            'BodyTextBold',
            parent=body_style,
            fontName='Helvetica-Bold'
        )

        meta_label_style = ParagraphStyle(
            'MetaLabel',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=colors.HexColor("#475569")
        )

        meta_val_style = ParagraphStyle(
            'MetaVal',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=9,
            textColor=colors.HexColor("#0f172a"),
            leading=11
        )

        disclaimer_style = ParagraphStyle(
            'DisclaimerCustom',
            parent=styles['Italic'],
            fontName='Helvetica-Oblique',
            fontSize=9,
            textColor=colors.HexColor("#64748b"),
            leading=13,
            spaceBefore=5,
            spaceAfter=5
        )

        story = []

        # ==========================================
        # 1. COVER PAGE
        # ==========================================
        story.append(Spacer(1, 1*inch))
        
        # Visual Cyan Banner Top
        story.append(Paragraph("<font color='#00f0ff'>█</font> VERIDOC FORENSICS", subtitle_style))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("DIGITAL MEDIA FORENSIC ANALYSIS REPORT", title_style))
        story.append(Paragraph("Consolidated Authenticity & Forensic Signal Integrity Verification", subtitle_style))
        
        story.append(Spacer(1, 1.5*inch))
        
        # Cover Metadata Table
        cover_meta = [
            [Paragraph("REPORT UUID:", meta_label_style), Paragraph(report_data["report_uuid"], meta_val_style)],
            [Paragraph("EVIDENCE ID:", meta_label_style), Paragraph(str(report_data["evidence_id"]), meta_val_style)],
            [Paragraph("GENERATED TIMESTAMP:", meta_label_style), Paragraph(report_data["generated_at"], meta_val_style)],
            [Paragraph("ORIGINAL FILENAME:", meta_label_style), Paragraph(report_data["evidence"]["original_filename"], meta_val_style)]
        ]
        
        t_cover = Table(cover_meta, colWidths=[2.0*inch, 4.5*inch])
        t_cover.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0,0), (-1,-1), 8),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
        ]))
        story.append(t_cover)
        story.append(PageBreak())

        # ==========================================
        # 2. SECTION 1 — Evidence Information
        # ==========================================
        story.append(Paragraph("Section 1 — Evidence Information", h1_style))
        story.append(Spacer(1, 0.1*inch))

        # Integrity Status banner
        integrity_status = report_data["evidence_integrity_status"]
        status_color = green_tint if integrity_status == "VERIFIED" else red_tint
        banner_text = f"<b>INTEGRITY STATUS: {integrity_status}</b>"
        
        t_banner = Table([[Paragraph(banner_text, ParagraphStyle('BannerText', parent=body_style, fontName='Helvetica-Bold', textColor=colors.white, fontSize=11, alignment=1))]], colWidths=[6.5*inch])
        t_banner.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), status_color),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t_banner)
        story.append(Spacer(1, 0.15*inch))

        # Detail Table
        size_mb = report_data["evidence"]["file_size"] / (1024 * 1024)
        evidence_meta = [
            [Paragraph("Original Filename", meta_label_style), Paragraph(report_data["evidence"]["original_filename"], meta_val_style)],
            [Paragraph("Media Category", meta_label_style), Paragraph(report_data["evidence"]["media_category"], meta_val_style)],
            [Paragraph("MIME Type", meta_label_style), Paragraph(report_data["evidence"]["mime_type"], meta_val_style)],
            [Paragraph("File Size", meta_label_style), Paragraph(f"{size_mb:.2f} MB ({report_data['evidence']['file_size']} bytes)", meta_val_style)],
            [Paragraph("Upload Timestamp", meta_label_style), Paragraph(report_data["evidence"]["uploaded_at"], meta_val_style)],
            [Paragraph("SHA-256 Signature", meta_label_style), Paragraph(report_data["evidence"]["sha256_hash"], meta_val_style)],
        ]
        t_evidence = Table(evidence_meta, colWidths=[2.0*inch, 4.5*inch])
        t_evidence.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('PADDING', (0,0), (-1,-1), 6),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
        ]))
        story.append(t_evidence)
        story.append(Spacer(1, 0.3*inch))

        # ==========================================
        # 3. SECTION 2 — Media Preprocessing
        # ==========================================
        story.append(Paragraph("Section 2 — Media Preprocessing", h1_style))
        
        pre_data = report_data["preprocessing"]
        pre_status = pre_data["status"]
        
        pre_meta = [
            [Paragraph("Preprocessing Status", meta_label_style), Paragraph(pre_status, meta_val_style)],
            [Paragraph("Face Detector", meta_label_style), Paragraph(pre_data["face_detector"], meta_val_style)],
            [Paragraph("Detected Faces", meta_label_style), Paragraph(str(pre_data["detected_faces"]), meta_val_style)],
        ]
        
        if pre_data.get("video_metadata"):
            v_meta = pre_data["video_metadata"]
            pre_meta.extend([
                [Paragraph("Resolution", meta_label_style), Paragraph(f"{v_meta['width']}x{v_meta['height']}", meta_val_style)],
                [Paragraph("FPS", meta_label_style), Paragraph(str(v_meta['fps']), meta_val_style)],
                [Paragraph("Duration", meta_label_style), Paragraph(f"{v_meta['duration_seconds']}s", meta_val_style)],
                [Paragraph("Sampled Frames", meta_label_style), Paragraph(str(v_meta['sampled_frames']), meta_val_style)],
            ])
            
        t_pre = Table(pre_meta, colWidths=[2.0*inch, 4.5*inch])
        t_pre.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('PADDING', (0,0), (-1,-1), 6),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
        ]))
        story.append(t_pre)
        story.append(Spacer(1, 0.3*inch))

        # ==========================================
        # 4. SECTION 3 — Face Detection
        # ==========================================
        story.append(Paragraph("Section 3 — Face Detection", h1_style))
        
        faces_list = pre_data.get("faces", [])
        if faces_list:
            # Face Coordinates table
            table_header_style = ParagraphStyle('HeadTable', parent=meta_label_style, textColor=colors.HexColor("#0f172a"))
            face_rows = [[
                Paragraph("<b>Face ID</b>", table_header_style),
                Paragraph("<b>Frame</b>", table_header_style),
                Paragraph("<b>Timestamp</b>", table_header_style),
                Paragraph("<b>Bounding Box (X, Y, W, H)</b>", table_header_style)
            ]]
            
            for face in faces_list:
                frame_idx = str(face["frame_index"]) if face["frame_index"] is not None else "N/A"
                timestamp = f"{face['timestamp_seconds']}s" if face["timestamp_seconds"] is not None else "N/A"
                bbox = f"({face['x']}, {face['y']}, {face['width']}, {face['height']})"
                
                face_rows.append([
                    Paragraph(f"Face #{face['face_index']}", body_style),
                    Paragraph(frame_idx, body_style),
                    Paragraph(timestamp, body_style),
                    Paragraph(bbox, body_style)
                ])
                
            t_faces = Table(face_rows, colWidths=[1.2*inch, 1.2*inch, 1.3*inch, 2.8*inch])
            t_faces.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
                ('PADDING', (0,0), (-1,-1), 6),
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
            ]))
            story.append(t_faces)
            story.append(Spacer(1, 0.15*inch))
            
            # Append Crops Thumbnails in a grid structure
            # Store up to 6 crop images in report to keep doc size reasonable
            crop_thumbnails = []
            row_cells = []
            for idx, face in enumerate(faces_list[:6]):
                crop_filename = face["crop_path"]
                crop_physical_path = settings.FACE_CROPS_DIR / crop_filename
                
                if crop_physical_path.exists():
                    try:
                        img_flow = Image(str(crop_physical_path), width=1.0*inch, height=1.0*inch)
                        lbl = Paragraph(f"Face #{face['face_index']}", ParagraphStyle('ImgLbl', parent=body_style, fontSize=8, alignment=1))
                        cell = Table([[img_flow], [lbl]], colWidths=[1.2*inch])
                        cell.setStyle(TableStyle([
                            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                            ('PADDING', (0,0), (-1,-1), 2),
                        ]))
                        row_cells.append(cell)
                        
                        if len(row_cells) == 4:
                            crop_thumbnails.append(row_cells)
                            row_cells = []
                    except Exception as e:
                        pass
                        
            if row_cells:
                # Pad remaining columns to align
                while len(row_cells) < 4:
                    row_cells.append("")
                crop_thumbnails.append(row_cells)
                
            if crop_thumbnails:
                story.append(Paragraph("<b>Extracted Face Crop Slices:</b>", body_style))
                t_thumbnails = Table(crop_thumbnails, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                t_thumbnails.setStyle(TableStyle([
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ]))
                story.append(t_thumbnails)
        else:
            story.append(Paragraph("No faces detected in the preprocessed evidence file.", body_style))
            
        story.append(PageBreak())

        # ==========================================
        # 5. SECTION 4 — Deepfake Analysis
        # ==========================================
        story.append(Paragraph("Section 4 — Deepfake Analysis", h1_style))
        
        df_data = report_data["deepfake"]
        df_status = df_data["status"]
        
        df_meta = [
            [Paragraph("Model", meta_label_style), Paragraph(df_data["model_name"], meta_val_style)],
            [Paragraph("Model Version", meta_label_style), Paragraph(df_data["model_version"], meta_val_style)],
            [Paragraph("Model Status", meta_label_style), Paragraph(df_data["model_status"], meta_val_style)],
            [Paragraph("Overall Result", meta_label_style), Paragraph(df_data["overall_result"], meta_val_style)],
            [Paragraph("Overall Confidence", meta_label_style), Paragraph(df_data["overall_confidence"], meta_val_style)]
        ]
        
        t_df = Table(df_meta, colWidths=[2.0*inch, 4.5*inch])
        t_df.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('PADDING', (0,0), (-1,-1), 6),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
        ]))
        story.append(t_df)
        story.append(Spacer(1, 0.15*inch))
        
        # Notice box
        if df_data["model_status"] == "MODEL_NOT_CONFIGURED":
            warning_text = "<b>No neural authenticity classification was performed because validated model weights are not configured.</b>"
            t_notice = Table([[Paragraph(warning_text, ParagraphStyle('NoticeTxt', parent=body_style, textColor=colors.HexColor("#b45309"), fontSize=9.5))]], colWidths=[6.5*inch])
            t_notice.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fef3c7")),
                ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#fde68a")),
                ('PADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(t_notice)
            
        story.append(Spacer(1, 0.3*inch))

        # ==========================================
        # 6. SECTION 5 — ELA Analysis
        # ==========================================
        story.append(Paragraph("Section 5 — Error Level Analysis (ELA)", h1_style))
        
        ela_data = report_data["ela"]
        ela_status = ela_data["status"]
        
        if ela_status == "COMPLETED":
            ela_meta = [
                [Paragraph("ELA Status", meta_label_style), Paragraph("COMPLETED", meta_val_style)],
                [Paragraph("JPEG Target Quality", meta_label_style), Paragraph(f"{ela_data['jpeg_quality']} / 100", meta_val_style)],
                [Paragraph("Mean Error Level", meta_label_style), Paragraph(str(ela_data['mean_error']), meta_val_style)],
                [Paragraph("Maximum Error Level", meta_label_style), Paragraph(str(ela_data['max_error']), meta_val_style)],
                [Paragraph("Error Level Variance", meta_label_style), Paragraph(str(ela_data['error_variance']), meta_val_style)],
            ]
            
            t_ela_meta = Table(ela_meta, colWidths=[2.2*inch, 4.3*inch])
            t_ela_meta.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
                ('PADDING', (0,0), (-1,-1), 6),
                ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
            ]))
            story.append(t_ela_meta)
            story.append(Spacer(1, 0.15*inch))
            
            # ELA Visualization flowable
            ela_filename = ela_data["visualization_path"]
            ela_physical_path = settings.ELA_DIR / ela_filename
            if ela_physical_path.exists():
                try:
                    ela_img = Image(str(ela_physical_path), width=3.5*inch, height=2.5*inch)
                    t_ela_img = Table([[ela_img]], colWidths=[6.5*inch])
                    t_ela_img.setStyle(TableStyle([
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('PADDING', (0,0), (-1,-1), 4),
                    ]))
                    story.append(t_ela_img)
                except Exception as e:
                    pass
        else:
            story.append(Paragraph(f"ELA Status: {ela_status}", body_bold))
            
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "<b>Forensic ELA Signal Guide:</b> ELA represents pixel-level differences produced by controlled JPEG recompression. "
            "The resulting signal may be influenced by compression history, format conversion, editing, or other processing. "
            "It does not independently establish image manipulation.",
            disclaimer_style
        ))
        
        story.append(PageBreak())

        # ==========================================
        # 7. SECTION 6 — FFT Analysis
        # ==========================================
        story.append(Paragraph("Section 6 — Fast Fourier Transform (FFT) Analysis", h1_style))
        
        fft_data = report_data["fft"]
        fft_status = fft_data["status"]
        
        if fft_status == "COMPLETED":
            fft_meta = [
                [Paragraph("FFT Status", meta_label_style), Paragraph("COMPLETED", meta_val_style)],
                [Paragraph("High-Frequency Radius", meta_label_style), Paragraph(f"{fft_data['high_freq_radius_percent']}% of Diagonal Mask", meta_val_style)],
                [Paragraph("Average Magnitude", meta_label_style), Paragraph(str(fft_data['average_magnitude']), meta_val_style)],
                [Paragraph("High-Frequency Energy Ratio", meta_label_style), Paragraph(f"{fft_data['high_frequency_ratio'] * 100:.3f}%", meta_val_style)],
                [Paragraph("Spectral Entropy Coefficient", meta_label_style), Paragraph(str(fft_data['spectral_entropy']), meta_val_style)],
            ]
            
            t_fft_meta = Table(fft_meta, colWidths=[2.2*inch, 4.3*inch])
            t_fft_meta.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
                ('PADDING', (0,0), (-1,-1), 6),
                ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
            ]))
            story.append(t_fft_meta)
            story.append(Spacer(1, 0.15*inch))
            
            # FFT Visualization flowable
            fft_filename = fft_data["visualization_path"]
            fft_physical_path = settings.FFT_DIR / fft_filename
            if fft_physical_path.exists():
                try:
                    fft_img = Image(str(fft_physical_path), width=3.0*inch, height=3.0*inch)
                    t_fft_img = Table([[fft_img]], colWidths=[6.5*inch])
                    t_fft_img.setStyle(TableStyle([
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('PADDING', (0,0), (-1,-1), 4),
                    ]))
                    story.append(t_fft_img)
                except Exception as e:
                    pass
        else:
            story.append(Paragraph(f"FFT Status: {fft_status}", body_bold))

        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "<b>Forensic FFT Signal Guide:</b> FFT metrics describe the frequency-domain characteristics of the analyzed image. "
            "They are statistical signal features and do not independently establish authenticity or manipulation.",
            disclaimer_style
        ))
        
        story.append(PageBreak())

        # ==========================================
        # 8. SECTION 7 — Audit Trail (Custody Chain)
        # ==========================================
        story.append(Paragraph("Section 7 — Custody Chain / Audit Trail", h1_style))
        story.append(Spacer(1, 0.05*inch))
        
        audit_list = report_data["audit_logs"]
        if audit_list:
            audit_header = ParagraphStyle('HeadAuditTable', parent=meta_label_style, textColor=colors.HexColor("#0f172a"))
            audit_rows = [[
                Paragraph("<b>Timestamp (UTC)</b>", audit_header),
                Paragraph("<b>Audit Action Event</b>", audit_header),
                Paragraph("<b>Custody Logs Description Details</b>", audit_header)
            ]]
            
            for log in audit_list:
                audit_rows.append([
                    Paragraph(log["timestamp"], body_style),
                    Paragraph(log["action"], body_bold),
                    Paragraph(log["details"], body_style)
                ])
                
            t_audit = Table(audit_rows, colWidths=[1.8*inch, 1.8*inch, 2.9*inch])
            t_audit.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
                ('PADDING', (0,0), (-1,-1), 6),
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f8fafc")),
            ]))
            story.append(t_audit)
        else:
            story.append(Paragraph("No audit logs matching this evidence record are registered.", body_style))
            
        story.append(Spacer(1, 0.4*inch))

        # ==========================================
        # 9. SECTION 8 — Forensic Interpretation
        # ==========================================
        story.append(Paragraph("Section 8 — Forensic Interpretation Summary", h1_style))
        story.append(Spacer(1, 0.05*inch))
        
        interpretation_text = (
            "The submitted media successfully underwent SHA-256 integrity verification, preprocessing, face localization, "
            "and independent ELA and FFT signal analysis. The ELA and FFT outputs represent statistical image-signal "
            "characteristics and should not independently be interpreted as proof of manipulation. Neural deepfake classification "
            "was not performed because the configured MesoInception-4 model weights are unavailable. Therefore, this report does "
            "not provide a definitive REAL/FAKE classification."
        )
        
        t_summary = Table([[Paragraph(interpretation_text, ParagraphStyle('SummaryTxt', parent=body_style, fontSize=10, leading=15))]], colWidths=[6.5*inch])
        t_summary.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
            ('PADDING', (0,0), (-1,-1), 12),
        ]))
        story.append(t_summary)

        # Build Document
        doc.build(story)
