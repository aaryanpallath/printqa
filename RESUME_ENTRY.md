# printqa — Automated Print-Quality Inspection
Tech: Python, OpenCV, NumPy, Matplotlib, pytest, GitHub Actions
- Built an inspection tool that aligns scanned print test charts with OpenCV and measures streaks, banding, dead nozzles, color registration and density against a YAML spec, producing CSV and HTML trend reports for a print run.
- Validated it on 530 simulated scans with known defects: zero false alarms on 110 clean pages, all 150 dead nozzles found exactly, and color registration measured to 0.9 µm RMSE with 85 µm scanner pixels.
- Traced a 400 µm registration error to threshold bias on slightly tilted lines and replaced it with arm-profile centroids; flagged each defect in a 60-page degrading run within one page of it going out of spec.
