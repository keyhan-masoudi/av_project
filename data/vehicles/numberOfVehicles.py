import xml.etree.ElementTree as ET
import csv
import os


def count_vehicles_per_timestep(xml_file, csv_file):
    """
    یک فایل XML (خروجی FCD) را پارس می‌کند و تعداد وسایل نقلیه
    در هر تایم‌استپ را در یک فایل CSV می‌نویسد و میانگین را محاسبه می‌کند.
    """

    # بررسی اینکه آیا فایل ورودی اصلا وجود دارد
    if not os.path.exists(xml_file):
        print(f"خطا: فایل '{xml_file}' پیدا نشد.")
        print("لطفاً مطمئن شوید فایل XML در همین پوشه قرار دارد یا آدرس کامل آن را وارد کنید.")
        return

    print(f"شروع پردازش فایل: {xml_file} ...")

    # --- اضافه شده ---
    # متغیرهایی برای محاسبه میانگین
    total_vehicle_counts = 0  # مجموع کل شمارش‌ها
    total_timesteps = 0       # تعداد کل تایم‌استپ‌ها
    # --- پایان بخش اضافه شده ---

    try:
        # پارس کردن درخت XML
        context = ET.iterparse(xml_file, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            # ایجاد یک نویسنده CSV
            writer = csv.writer(f)
            # نوشتن ردیف هدر (سرتیتر)
            writer.writerow(['time', 'vehicle_count'])

            vehicle_count = 0
            current_time = None

            # پیمایش تمام المنت‌ها در فایل
            for event, elem in context:
                # وقتی به شروع تگ <timestep> می‌رسیم
                if event == 'start' and elem.tag == 'timestep':
                    current_time = elem.get('time')
                    vehicle_count = 0  # شمارنده را برای این تایم‌استپ صفر می‌کنیم

                # اگر در داخل یک تایم‌استپ هستیم و به تگ <vehicle> رسیدیم
                elif event == 'start' and elem.tag == 'vehicle' and current_time is not None:
                    vehicle_count += 1

                # وقتی به انتهای تگ <timestep> می‌رسیم
                elif event == 'end' and elem.tag == 'timestep':
                    # داده‌های این تایم‌استپ را می‌نویسیم
                    if current_time is not None:
                        writer.writerow([current_time, vehicle_count])

                        # --- اضافه شده ---
                        # به‌روزرسانی مقادیر برای محاسبه میانگین
                        total_vehicle_counts += vehicle_count
                        total_timesteps += 1
                        # --- پایان بخش اضافه شده ---

                    # حافظه را برای المان‌های پردازش شده آزاد می‌کنیم
                    root.clear()
                    current_time = None  # ریست می‌کنیم

        print(f"\nپردازش با موفقیت انجام شد.")
        print(f"نتایج در فایل '{csv_file}' ذخیره گردید.")

        # --- اضافه شده ---
        # محاسبه و چاپ میانگین پس از اتمام حلقه
        if total_timesteps > 0:
            average_vehicles = total_vehicle_counts / total_timesteps
            print(f"\n--- آمار ---")
            print(f"تعداد کل تایم‌استپ‌های پردازش شده: {total_timesteps}")
            print(f"مجموع کل شمارش وسایل نقلیه: {total_vehicle_counts}")
            print(f"میانگین وسایل نقلیه در هر تایم‌استپ: {average_vehicles:.2f}")
        else:
            print("\nهیچ تایم‌استپی برای محاسبه میانگین پیدا نشد.")
        # --- پایان بخش اضافه شده ---

    except ET.ParseError as e:
        print(f"خطا در پارس کردن فایل XML: {e}")
    except FileNotFoundError:
        print(f"خطا: فایل {xml_file} پیدا نشد.")
    except IOError as e:
        print(f"خطا هنگام نوشتن در فایل CSV: {e}")
    except Exception as e:
        print(f"یک خطای غیرمنتظره رخ داد: {e}")


if __name__ == "__main__":

    input_xml = "chunk_0.xml"

    default_csv = os.path.splitext(input_xml)[0] + "_counts.csv"

    output_csv = "numberOfVehicles"

    if not output_csv:
        output_csv = default_csv

    count_vehicles_per_timestep(input_xml, output_csv)