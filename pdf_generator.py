from fpdf import FPDF
import datetime

class CareReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Clinical Care Coordination Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')

def sanitize(obj):
    if isinstance(obj, dict):
        return "\n".join([f"{str(k).replace('_', ' ').title()}: {sanitize(v)}" for k, v in obj.items()])
    elif isinstance(obj, list):
        return "\n".join([f"* {sanitize(v)}" for v in obj])
    return str(obj).encode('latin-1', 'ignore').decode('latin-1')

def generate_pdf(report_data, patient_data, prediction):
    pdf = CareReportPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Patient Context & Risk Assessment", 0, 1)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Patient Age: {patient_data['Age']} | Gender: {patient_data['Gender']}", 0, 1)
    pdf.cell(0, 8, f"Predicted No-Show Probability: {prediction:.2%}", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Executive Summary", 0, 1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, sanitize(report_data.get('summary', '')))
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Key Contributing Factors", 0, 1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, sanitize(report_data.get('factors', [])))
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Recommended Interventions", 0, 1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, sanitize(report_data.get('strategies', [])))
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "4. Supporting Sources & Disclaimers", 0, 1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, sanitize(report_data.get('sources', [])))
    pdf.ln(2)
    
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, sanitize(report_data.get('disclaimers', 'N/A')))

    return bytes(pdf.output())
