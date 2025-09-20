import cv2
import numpy as np
import pandas as pd
import os

# Load answer key
answer_key = pd.read_csv("Innomatics/Trail-1.2/answer_key.csv")
answers = {}
for col in answer_key.columns:
    for cell in answer_key[col]:
        cell = str(cell).strip()
        if '-' in cell:
            q, a = [x.strip() for x in cell.split('-', 1)]
            try:
                qnum = int(q.replace('.', '').replace(' ', ''))
                answers[qnum] = a.lower()
            except:
                continue

# Parameters and paths
input_folder = "Innomatics/Trail-1.2/omr_sheets"
output_folder = "Innomatics/Trail-1.2/results"
os.makedirs(output_folder, exist_ok=True)
options = ['a', 'b', 'c', 'd', 'e']  # adjust as per OMR options
min_contour_area = 500
row_threshold = 15  # vertical proximity for grouping contours in the same row

def group_contours_by_rows(contours, threshold):
    rows = []
    sortedContours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    current_row = []
    last_y = None

    for c in sortedContours:
        x, y, w, h = cv2.boundingRect(c)
        if last_y is None or abs(y - last_y) <= threshold:
            current_row.append(c)
            last_y = y
        else:
            rows.append(current_row)
            current_row = [c]
            last_y = y
    if current_row:
        rows.append(current_row)
    return rows

def evaluate_omr(image_path, student_id):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blurred, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = [c for c in cnts if cv2.contourArea(c) > min_contour_area]

    rows = group_contours_by_rows(bubble_contours, row_threshold)

    student_result = {"Student_ID": student_id, "Score": 0}
    question_num = 1

    for row in rows:
        row = sorted(row, key=lambda c: cv2.boundingRect(c)[0])
        # Process in groups of option bubbles per question
        for i in range(0, len(row), len(options)):
            group = row[i:i+len(options)]
            if len(group) < len(options):
                continue

            filled_option = None
            max_pixels = 0
            for idx, cont in enumerate(group):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [cont], -1, 255, -1)
                pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                if pixels > max_pixels:
                    max_pixels = pixels
                    filled_option = options[idx]

            correct_answer = answers.get(question_num, None)
            correct_answer = correct_answer.lower() if correct_answer else None
            is_correct = (filled_option == correct_answer)
            if is_correct:
                student_result["Score"] +=1

            student_result[f"Q{question_num}_Marked"] = filled_option
            student_result[f"Q{question_num}_Correct"] = correct_answer
            student_result[f"Q{question_num}_Result"] = "Correct" if is_correct else "Wrong"

            # Annotate image for visual feedback
            if correct_answer in options:
                opt_idx = options.index(correct_answer)
                if opt_idx < len(group):
                    x, y, w, h = cv2.boundingRect(group[opt_idx])
                    color = (0, 255, 0) if is_correct else (0, 0, 255)
                    cv2.putText(image, 
                                "Correct" if is_correct else "Wrong",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2)

            question_num +=1

    save_path = os.path.join(output_folder, f"{student_id}_checked.png")
    cv2.imwrite(save_path, image)
    return student_result

def main():
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]

    results = []
    for file in image_files:
        path = os.path.join(input_folder, file)
        student_id = os.path.splitext(file)[0]
        print(f"Evaluating {student_id} ...")
        res = evaluate_omr(path, student_id)
        if res:
            results.append(res)

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_folder, "all_students_results.csv")
    df.to_csv(csv_path, index=False)
    print("Evaluation complete. Check result CSV and annotated images in 'results' folder.")

if __name__ == "__main__":
    main()
