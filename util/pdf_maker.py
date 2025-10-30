import re
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

from util.path import REPORT_DIR

def make_pdf_and_save(md_content: str):
    
    # 한글 폰트 등록
    pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 스타일 정의
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='KoreanH1',
        fontName='HYSMyeongJo-Medium',
        fontSize=18,
        leading=22,
        spaceAfter=12,
        textColor='#2c3e50'
    ))
    styles.add(ParagraphStyle(
        name='KoreanH2',
        fontName='HYSMyeongJo-Medium',
        fontSize=14,
        leading=18,
        spaceAfter=10,
        textColor='#34495e'
    ))
    styles.add(ParagraphStyle(
        name='KoreanH3',
        fontName='HYSMyeongJo-Medium',
        fontSize=12,
        leading=16,
        spaceAfter=8,
        textColor='#7f8c8d'
    ))
    styles.add(ParagraphStyle(
        name='KoreanNormal',
        fontName='HYSMyeongJo-Medium',
        fontSize=10,
        leading=14,
        spaceAfter=6
    ))

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = str(REPORT_DIR / f"samsung_ai_report_{now}.pdf")
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    story = []

    # 마크다운 파싱
    for line in md_content.split('\n'):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.1*inch))
        elif line.startswith('# '):
            story.append(Paragraph(line[2:], styles['KoreanH1']))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], styles['KoreanH2']))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], styles['KoreanH3']))
        elif line.startswith('---'):
            story.append(Spacer(1, 0.2*inch))
        else:
            # 볼드 처리
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            story.append(Paragraph(line, styles['KoreanNormal']))

    doc.build(story)
    
    print(f"PDF 생성 완료 - {file_path}")
    
    return file_path