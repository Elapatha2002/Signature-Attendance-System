# Student Attendance Management System (CS402.3)

A Python prototype that reads photographed signing sheets, decides whether each
student signed, stores the attendance in SQLite and visualizes it.

## Main technologies

- Python 3.10+
- **OpenCV** - CLAHE contrast normalisation, adaptive thresholding, Hough skew
  estimation, morphological line extraction, contour analysis, perspective
  correction, connected-component filtering and ORB features
- **XML** (`xml.etree.ElementTree`) - student, subject, session and threshold data
- **SQLite** (`sqlite3`) - local attendance storage
- **Matplotlib** - attendance charts and report figures
- **scikit-image** - structural similarity for the signature experiment
  (optional; a built-in SSIM implementation is used when it is not installed)

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

| Command | Purpose |
| --- | --- |
| `python sams.py input_signing_sheets/sheet1.jpeg info.xml` | Process one sheet |
| `python run_all.py --reset` | Process all five sheets and print the attendance matrix |
| `python infovis.py 10009301` | Attendance chart for one student |
| `python investigate.py 10000409` | Calibrated signature-consistency check for one student |
| `python audit_sheet.py` | One combined audit grid: every student x every lecture |
| `python robustness.py` | Measure sensitivity to image degradation |
| `python report_figures.py` | Regenerate every figure used in the report |
| `pytest -q` | Run the test suite |

`sams.py` prints its progress through the nine processing stages and writes an
image for each one under `output/<sheet name>/`.

## Processing pipeline

1. Load the photograph and resize it to a 1400 px working width.
2. Convert to greyscale.
3. Normalise local contrast with CLAHE, then binarise with an adaptive mean
   threshold.
4. Estimate the page skew from the printed rules with a probabilistic Hough
   transform and rotate the page level.
5. Extract long horizontal and vertical rules with morphological opening.
6. Find the table contour, approximate its four corners and remove perspective
   with a projective transform.
7. **Measure** the printed grid in the corrected table and take the row and
   column bounds from it. The student block is identified as the run of
   separators with uniform row spacing, so a merged date table does not shift
   the rows.
8. Build a colour-ink mask in HSV and a dark-ink mask by removing the printed
   rules, for each signature cell.
9. Apply the threshold rule, write the result to SQLite and save an annotated
   image.

## Results on the five supplied sheets

The 30 signature cells contain 25 signed cells and 5 absences, and all 30 match
the manually read labels. Four of the absences are empty cells. The fifth is
different: on sheet 2 staff recorded the absence of student 10009306 by writing
a struck-out `-ab-` mark into the cell, so the cell contains ink and an
ink-ratio rule alone would call it present. That case is handled separately, as
described below.

For the cells decided by ink ratio the margin is wide: the strongest blank cell
reaches only 0.65x the threshold while the weakest signed cell reaches 3.71x.

## Written absence marks

A cell holding a struck-out mark such as `-ab-` is recognised by geometry rather
than by reading it. A strike stroke is a component that is long and thin, and in
a struck-out mark it sits *beside* the written token at the same height. That is
what separates it from a signature that happens to be underlined, where the rule
sits *below* the writing and overlaps it horizontally.

This rule is calibrated on a single example, because the coursework dataset
contains exactly one written absence mark. It is correct on the five supplied
photographs, but `robustness.py` shows it is the most fragile part of the
pipeline: heavy degradation fragments the strike stroke and the mark is then
read as a signature. More samples would be needed before relying on it.

## Robustness

`robustness.py` re-runs the same sheets under 17 synthetic degradations. The
signature and blank-cell decisions stay correct under rotation up to +/-10
degrees, perspective tilt up to 15%, downscaling to 20%, blur, +30%/-35%
exposure, sensor noise and JPEG quality 10. The residual errors are the written
absence mark described above, plus one blank cell under a 15 px Gaussian blur.

The signature comparison is deliberately reported as a weak signal: measured
over 41 genuine and 259 impostor pairs it reaches an ROC area of about 0.82, so
it is only useful for prioritising manual review.

## Signature audit sheet

`audit_sheet.py` produces one grid covering the whole class: a row per student,
a column per lecture, and in each cell the signature that was extracted from
that sheet together with a verdict.

Each signature is compared with the same student's other signatures and the
median score is kept, so one unusual sample cannot condemn the rest. That median
is read against the genuine and impostor score distributions measured across the
class, giving the posterior probability that the sample resembles a different
writer. Cells are marked *not suspicious*, *borderline* or *suspicious*.

Two safeguards keep the output honest. The ROC area is printed in the title, so
the number can never be quoted without its uncertainty. And when *every* sample
of one student is flagged, the sheet says so explicitly: that means the
reference set itself is inconsistent, which points at capture variation, for
example a signature that overruns its cell and is clipped differently on each
sheet, rather than at one suspect signature.

A flagged cell means "a human should look at this", never "this is a forgery".

## Folder structure

```text
Prototype/
|-- attendance/                  Core package
|   |-- database.py              SQLite schema and queries
|   |-- image_processor.py       The image-processing pipeline
|   |-- models.py                Typed data classes
|   |-- signature_matcher.py     Signature similarity and threshold calibration
|   |-- visualizer.py            Matplotlib charts
|   `-- xml_loader.py            info.xml parsing and validation
|-- data/attendance.db           Generated database
|-- input_signing_sheets/        The five supplied photographs
|-- output/                      Stage images, charts and report figures
|-- tests/                       Unit and regression tests
|-- info.xml                     Students, sessions and thresholds
|-- sams.py                      Process one sheet
|-- run_all.py                   Process every session
|-- infovis.py                   Student attendance chart
|-- investigate.py               Signature investigation
|-- robustness.py                Degradation sensitivity harness
|-- requirements.txt
`-- README.md
```

## Configuration

`info.xml` holds the students, the session dates and the processing thresholds,
so none of them require a code change:

```xml
<processing signatureColumnStart="0.79"
            colorRatioThreshold="0.012"
            residualRatioThreshold="0.020" />
```

`signatureColumnStart` is only a fallback. When the printed grid is found, the
signature column is measured from it instead.
