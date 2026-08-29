import pandas as pd


def calculate_soft_metrics(file_path):

    df = pd.read_csv(file_path)

    # Select only soft tasks
    soft_tasks = df[
        df["task_id"].astype(str).str.contains("_S_")
    ].copy()

    # Sum deadline difference
    total_deadline_diff = soft_tasks["deadline_diff"].sum()

    # Calculate tardiness
    soft_tasks["tardiness"] = (
        soft_tasks["finish_time"] - soft_tasks["deadline"]
    ).clip(lower=0)

    total_tardiness = soft_tasks["tardiness"].sum()

    return (
        total_deadline_diff,
        total_tardiness,
        len(soft_tasks)
    )


file1 = r"D:\code\final_av\av_project\Results_Success\success_deadlines_report_DDPGNew_First Choice_-1_1_2_Melbourne_withHardTasks_withoutParallel.csv"
file2 = r"D:\code\final_av\av_project\Results\missed_deadlines_report_DDPGNew_First Choice_-1_1_2_Melbourne_withHardTasks_withoutParallel.csv"

# file1 = r"D:\code\final_av\av_project\Results_Success\success_deadlines_report_Greedy_First Choice_-1_1_2_Melbourne_withHardTasks_withoutParallel.csv"
# file2 = r"D:\code\final_av\av_project\Results\missed_deadlines_report_Greedy_First Choice_-1_1_2_Melbourne_withHardTasks_withoutParallel.csv"

diff1, tard1, count1 = calculate_soft_metrics(file1)
diff2, tard2, count2 = calculate_soft_metrics(file2)


print("========== File 1 ==========")
print(f"Soft tasks: {count1}")
print(f"Sum deadline diff: {diff1}")


print("\n========== File 2 ==========")
print(f"Soft tasks: {count2}")
print(f"Sum deadline diff: {diff2}")


print("\n========== Total ==========")
print(f"Soft tasks: {count1 + count2}")
print(f"Sum deadline diff: {diff1 + diff2}")