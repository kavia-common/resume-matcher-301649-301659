from typing import List, Tuple, Set
import io
import re

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# External parsers
from PyPDF2 import PdfReader  # lightweight and reliable for many PDFs
from docx import Document  # python-docx

# Create the FastAPI app with metadata
app = FastAPI(
    title="ATS Resume Matcher API",
    description="Parses resumes (PDF/DOCX), extracts key details, and compares against a job description to compute a keyword match score.",
    version="1.0.0",
    contact={"name": "ATS Resume Matcher"},
    license_info={"name": "MIT"},
)

# Allow frontend on 3000; also allow additional origins via env if desired
# Using permissive defaults for simplicity; narrow in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify explicit origins e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Models for OpenAPI and typing
# -----------------------------
class ExtractedData(BaseModel):
    name: str = Field("", description="Candidate's name if extracted")
    contact: str = Field("", description="Contact info (email/phone) if extracted")
    skills: List[str] = Field(default_factory=list, description="Detected skills as keywords")
    experience: str = Field("", description="Raw extracted experience text")
    education: str = Field("", description="Raw extracted education text")


class MatchResponse(BaseModel):
    score: float = Field(..., description="Match score between 0 and 100")
    matched_keywords: List[str] = Field(..., description="Keywords that appear in the resume")
    missing_keywords: List[str] = Field(..., description="Keywords from the job description that are missing in the resume")
    summary: str = Field(..., description="Short textual summary and suggestions")
    extracted: ExtractedData = Field(..., description="Extracted resume fields")


# --------------------------------
# Utility parsing and NLP functions
# --------------------------------

def _clean_text(text: str) -> str:
    """Normalize whitespace and lower case for matching."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF using PyPDF2 entirely in-memory."""
    try:
        with io.BytesIO(file_bytes) as fh:
            reader = PdfReader(fh)
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    # Continue even if a page fails
                    continue
            return "\n".join(pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}")


def _extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract raw text from a DOCX using python-docx entirely in-memory."""
    try:
        with io.BytesIO(file_bytes) as fh:
            doc = Document(fh)
            texts = []
            for para in doc.paragraphs:
                texts.append(para.text)
            # Include tables text as well
            for table in getattr(doc, "tables", []):
                for row in table.rows:
                    for cell in row.cells:
                        texts.append(cell.text)
            return "\n".join(filter(None, texts))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse DOCX: {e}")


def _extract_name(text: str) -> str:
    """Heuristic: Use first non-empty line that looks like a name (2-4 words, capitalized)."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip obvious headings
        if line.lower() in {"resume", "curriculum vitae", "cv"}:
            continue
        # Simple name heuristic: 2-4 tokens, mostly starting uppercase
        tokens = line.split()
        if 2 <= len(tokens) <= 4:
            caps = sum(1 for t in tokens if re.match(r"^[A-Z][a-zA-Z\-'.]*$", t))
            if caps >= max(2, len(tokens) - 1):
                return line
        # Fallback to first non-empty line early
        if len(line.split()) <= 8:
            return line
    return ""


def _extract_contact(text: str) -> str:
    """Find email and phone. Concatenate if both found."""
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"(\+?\d[\d\s().-]{7,}\d)", text)
    parts = []
    if email_match:
        parts.append(email_match.group(0))
    if phone_match:
        parts.append(phone_match.group(0))
    return " | ".join(parts)


def _section_text(text: str, start_keywords: List[str], stop_keywords: List[str]) -> str:
    """Extract text between start and next stop section headings (very heuristic)."""
    lower = text.lower()
    start_idx = -1
    for kw in start_keywords:
        idx = lower.find(kw)
        if idx != -1:
            if start_idx == -1 or idx < start_idx:
                start_idx = idx
    if start_idx == -1:
        return ""
    # find the next stop keyword after start_idx
    next_stop = len(text)
    for kw in stop_keywords:
        idx = lower.find(kw, start_idx + 1)
        if idx != -1 and idx < next_stop:
            next_stop = idx
    return text[start_idx:next_stop].strip()


def _extract_skills(text: str) -> List[str]:
    """Extract skills from a dedicated Skills section when possible, otherwise keyword pass."""
    section = _section_text(
        text,
        start_keywords=["skills", "technical skills", "core competencies"],
        stop_keywords=["experience", "work experience", "professional experience", "education", "projects", "summary", "certifications"],
    )
    candidates = []
    source = section if section else text
    # Split on common separators
    for token in re.split(r"[,\n;•/|]", source):
        token = token.strip()
        if not token:
            continue
        # Filter out too generic words
        if len(token) > 2 and not token.lower().startswith(("experience", "education", "summary")):
            candidates.append(token)
    # Normalize: lowercase, dedupe while preserving order
    seen = set()
    skills = []
    for c in candidates:
        norm = re.sub(r"[^a-z0-9+#.\-\s]", "", c.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        skills.append(norm)
    return skills[:100]


def _extract_experience(text: str) -> str:
    return _section_text(
        text,
        start_keywords=["experience", "work experience", "professional experience", "employment history"],
        stop_keywords=["education", "skills", "projects", "certifications", "summary"],
    )


def _extract_education(text: str) -> str:
    return _section_text(
        text,
        start_keywords=["education", "academic background"],
        stop_keywords=["experience", "work experience", "skills", "projects", "certifications", "summary"],
    )


def _tokenize_keywords(s: str) -> List[str]:
    """Tokenize a description into normalized keyword phrases."""
    # Split by punctuation and line breaks, then also consider n-grams by commas/semicolons
    raw = re.split(r"[\n;,/•|]", s)
    tokens = []
    for r in raw:
        r = r.strip().lower()
        if not r:
            continue
        # Further split long lines by ' - ' or ' • ' or parentheses closures
        parts = re.split(r"[-()]", r)
        for p in parts:
            p = re.sub(r"[^a-z0-9+#.\s]", " ", p)
            p = re.sub(r"\s+", " ", p).strip()
            if p and len(p) > 1:
                tokens.append(p)
    # Also add individual words for robustness (but filter short stopwords)
    words = [w for w in re.split(r"\W+", s.lower()) if len(w) > 2]
    tokens.extend(words)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:200]


def _keyword_sets_from_job_description(jd_text: str) -> Set[str]:
    tokens = _tokenize_keywords(jd_text)
    # Remove common stopwords/minor words
    stop = {
        "and", "the", "with", "for", "you", "are", "our", "your", "will", "have", "has", "this", "that",
        "from", "but", "not", "who", "what", "when", "where", "why", "how", "can", "able", "work",
        "team", "role", "position", "company", "skills", "experience", "education", "years",
        "required", "preferred", "plus", "etc", "using", "use", "good", "strong",
    }
    keywords = []
    for t in tokens:
        # Keep phrases likely to be skills/tech or meaningful terms
        if t in stop:
            continue
        if len(t) <= 2 and t not in {"c", "go", "qa"}:
            continue
        keywords.append(t)
    # Deduplicate
    return {k for k in keywords}


def _compute_match(resume_text: str, jd_text: str) -> Tuple[float, List[str], List[str]]:
    """Compute keyword overlap score."""
    resume_norm = _clean_text(resume_text).lower()
    jd_keywords = _keyword_sets_from_job_description(jd_text)
    matched = []
    for kw in jd_keywords:
        # Simple containment; could be improved with fuzzy matching in future
        if kw in resume_norm:
            matched.append(kw)
    missing = sorted(list(jd_keywords - set(matched)))
    # Score is percentage of JD keywords found
    score = 0.0
    if jd_keywords:
        score = round(100.0 * len(matched) / len(jd_keywords), 2)
    return score, sorted(matched), missing


# -----------------------------
# Routes
# -----------------------------

@app.get("/", summary="Health Check", tags=["Health"])
def health_check():
    """Health check endpoint to verify the service is running."""
    return {"message": "Healthy"}


# PUBLIC_INTERFACE
@app.post(
    "/match",
    response_model=MatchResponse,
    summary="Match resume against job description",
    tags=["Matching"],
)
async def match_resume(
    resume: UploadFile = File(..., description="The resume file in PDF or DOCX format"),
    job_description: str = Form(..., description="The job description text to match against"),
):
    """
    Parse an uploaded resume (PDF/DOCX), extract basic fields, and compare against the provided
    job description. Returns a score and feedback including matched/missing keywords.

    Parameters:
    - resume: multipart file upload (PDF or DOCX). Processed entirely in-memory.
    - job_description: job description text to extract keywords from.

    Returns:
    - JSON payload with:
        - score (0-100)
        - matched_keywords, missing_keywords
        - summary
        - extracted: { name, contact, skills[], experience, education }
    """
    if resume.content_type not in {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        # Some browsers may send generic octet-stream; do a fallback on filename
        "application/octet-stream",
    }:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or DOCX file.")

    data = await resume.read()
    filename = (resume.filename or "").lower()

    # Extract text based on file type
    text = ""
    if filename.endswith(".pdf") or resume.content_type == "application/pdf":
        text = _extract_text_from_pdf(data)
    elif filename.endswith(".docx") or resume.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = _extract_text_from_docx(data)
    else:
        # Try PDF first then DOCX as a fallback for octet-stream
        try:
            text = _extract_text_from_pdf(data)
        except HTTPException:
            text = _extract_text_from_docx(data)

    text = text or ""
    # Basic extracted fields
    name = _extract_name(text)
    contact = _extract_contact(text)
    skills = _extract_skills(text)
    experience = _extract_experience(text)
    education = _extract_education(text)

    # Compute keyword matching
    score, matched, missing = _compute_match(text, job_description or "")

    # Prepare summary and suggestions
    if missing:
        suggestions = f"Consider adding or emphasizing: {', '.join(missing[:10])}"  # limit length
    else:
        suggestions = "Great alignment with the job description."

    summary = f"Match score: {score}%. {suggestions}"

    return MatchResponse(
        score=score,
        matched_keywords=matched,
        missing_keywords=missing,
        summary=summary,
        extracted=ExtractedData(
            name=name,
            contact=contact,
            skills=skills,
            experience=experience,
            education=education,
        ),
    )
