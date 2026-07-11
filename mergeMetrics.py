import os
import glob
import pandas as pd


def process_metrics():
    metrics_folder = "Results_Metrics"
    success_folder = "Results_success"
    missed_folder = "Results"

    # cities = ["Hamburg", "Melbourne"]
    cities = ["Melbourne"]

    for city in cities:
        city_metrics_folder = os.path.join(metrics_folder)
        city_success_folder = os.path.join(success_folder)
        city_missed_folder = os.path.join(missed_folder)

        data_list = []

        # Update file pattern to drop the "New" prefix if it's no longer used in the generator
        file_pattern = os.path.join(city_metrics_folder, "final_metrics_summary_*.xlsx")
        files = glob.glob(file_pattern)

        for index, file in enumerate(files, 1):
            filename = os.path.basename(file)

            # Clean name extraction
            base_suffix = filename.replace("final_metrics_summary_", "").replace(".xlsx", "")
            clean_name = base_suffix

            print(f"[{index}/{len(files)}] 🔄 Processing combination: {clean_name}")
            print(f"  ├─ 📄 Reading and extracting last row from main file: {filename}")

            # Update extensions to .csv based on the new simulator output
            success_file_name = f"success_deadlines_report_{base_suffix}.csv"
            missed_file_name = f"missed_deadlines_report_{base_suffix}.csv"

            success_file = os.path.join(city_success_folder, success_file_name)
            missed_file = os.path.join(city_missed_folder, missed_file_name)

            # Parse filename for variables
            # New format: {ALGORITHM}_{METHOD}_{THRESHOLD}_{TRAFFIC}_{ATTENUATION}_{CITY}_{HARDTASKS}_{PARALLEL}
            parts = clean_name.split("_")

            # Extracting values based on fixed positions from the end
            parallel = parts[-1]
            hardTasks = parts[-2]
            city_name = parts[-3]
            attenuation = parts[-4]
            traffic = parts[-5]
            threshold = parts[-6]
            method = parts[-7]
            algorithm = "_".join(parts[:-7])

            if traffic == "0":
                mapped_traffic = "1"
            elif traffic == "1":
                mapped_traffic = "2"
            else:
                mapped_traffic = traffic

            try:
                # Read main metric file
                df = pd.read_excel(file)

                if 'timeStep' in df.columns:
                    last_step_row = df.loc[df['timeStep'].idxmax()].to_frame().T
                else:
                    last_step_row = df.iloc[[-1]].copy()

                sum_035 = df['Threshold_0.35_Count'].sum() if 'Threshold_0.35_Count' in df.columns else 0
                sum_045 = df['Threshold_0.45_Count'].sum() if 'Threshold_0.45_Count' in df.columns else 0
                sum_055 = df['Threshold_0.55_Count'].sum() if 'Threshold_0.55_Count' in df.columns else 0
                total_count = sum_035 + sum_045 + sum_055

                perc_035 = (sum_035 / total_count * 100) if total_count > 0 else 0
                perc_045 = (sum_045 / total_count * 100) if total_count > 0 else 0
                perc_055 = (sum_055 / total_count * 100) if total_count > 0 else 0

                last_step_row['Threshold_0.35_Count'] = sum_035
                last_step_row['Threshold_0.45_Count'] = sum_045
                last_step_row['Threshold_0.55_Count'] = sum_055
                last_step_row['Threshold_0.35_Percentage'] = perc_035
                last_step_row['Threshold_0.45_Percentage'] = perc_045
                last_step_row['Threshold_0.55_Percentage'] = perc_055

                # Calculate Success files
                sum_success = 0
                if os.path.exists(success_file):
                    print(f"  ├─ 🟢 Calculating deadline_diff sum from: {success_file_name}")
                    df_success = pd.read_csv(success_file)  # Changed to read_csv
                    if 'deadline_diff' in df_success.columns:
                        sum_success = df_success['deadline_diff'].sum()
                else:
                    print(f"  ├─ ⚠️ File not found (value set to 0): {success_file_name}")

                # Calculate Missed files
                sum_failure = 0
                if os.path.exists(missed_file):
                    print(f"  ├─ 🔴 Calculating deadline_diff sum from: {missed_file_name}")
                    df_missed = pd.read_csv(missed_file)  # Changed to read_csv
                    if 'deadline_diff' in df_missed.columns:
                        sum_failure = df_missed['deadline_diff'].sum()
                else:
                    print(f"  ├─ ⚠️ File not found (value set to 0): {missed_file_name}")

                # Insert new computed columns
                last_step_row['success'] = sum_success
                last_step_row['failure'] = sum_failure
                last_step_row['Tardiness'] = sum_success + sum_failure

                # Add base columns including the new configuration fields
                last_step_row['Algorithm'] = algorithm
                last_step_row['method'] = method
                last_step_row['cityName'] = city_name
                last_step_row['NoiseLevel'] = mapped_traffic
                last_step_row['AttenuationLevel'] = attenuation
                last_step_row['HardTasks'] = hardTasks
                last_step_row['Parallel'] = parallel

                # Sort columns
                base_cols = [
                    'Algorithm', 'method', 'cityName', 'NoiseLevel', 'AttenuationLevel',
                    'HardTasks', 'Parallel', 'success', 'failure', 'Tardiness'
                ]
                ordered_cols = base_cols + [col for col in last_step_row.columns if col not in base_cols]
                last_step_row = last_step_row[ordered_cols]

                data_list.append(last_step_row)
                print(f"  └─ ✔️ Successfully processed this combination.\n")

            except Exception as e:
                print(f"  └─ ❌ Error processing files for {clean_name}: {e}\n")

        print(f"⚙️ Merging data, sorting, and creating group spacings for {city}...")

        if data_list:
            final_df = pd.concat(data_list, ignore_index=True)

            final_df['NoiseLevel'] = pd.to_numeric(final_df['NoiseLevel'], errors='coerce')
            final_df['AttenuationLevel'] = pd.to_numeric(final_df['AttenuationLevel'], errors='coerce')

            # Sort by noise and attenuation (descending), then algorithm, method, hardTasks, and parallel (ascending)
            final_df = final_df.sort_values(
                by=['NoiseLevel', 'AttenuationLevel', 'Algorithm', 'method', 'HardTasks', 'Parallel'],
                ascending=[False, False, True, True, True, True]
            )

            rows_with_blanks = []
            prev_noise = None
            prev_attenuation = None

            for index, row in final_df.iterrows():
                curr_noise = row['NoiseLevel']
                curr_attenuation = row['AttenuationLevel']

                if prev_noise is not None and (curr_noise != prev_noise or curr_attenuation != prev_attenuation):
                    empty_row = pd.Series({col: None for col in final_df.columns})
                    rows_with_blanks.append(empty_row)

                rows_with_blanks.append(row)
                prev_noise = curr_noise
                prev_attenuation = curr_attenuation

            final_df_with_blanks = pd.DataFrame(rows_with_blanks)

            output_filename = f"FinalMetric{city}.xlsx"
            final_df_with_blanks.to_excel(output_filename, index=False)
            print(f"💾 ✅ Final file '{output_filename}' created and saved successfully!\n")
        else:
            print(f"⚠️  No data found to save for city {city}.\n")


if __name__ == "__main__":
    print("🚀 Script execution started...")
    process_metrics()
    print("🎉 All tasks completed successfully.")