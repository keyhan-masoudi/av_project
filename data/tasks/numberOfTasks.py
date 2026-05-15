import xml.etree.ElementTree as ET

# نام فایل XML خود را در اینجا قرار دهید
file_name = 'chunk_0.xml'

try:
    # فایل XML را تجزیه می‌کند
    tree = ET.parse(file_name)
    # ریشه (root) درخت XML را دریافت می‌کند
    root = tree.getroot()

    # پیدا کردن تمام المنت‌های 'task' در کل درخت
    tasks = root.findall('.//task')

    # شمارش تعداد تسک‌های پیدا شده
    total_tasks = len(tasks)

    # چاپ نتیجه
    print(f"مجموع تعداد تسک‌ها در فایل '{file_name}': {total_tasks}")

except FileNotFoundError:
    print(f"خطا: فایل '{file_name}' پیدا نشد. لطفاً مطمئن شوید فایل در همین پوشه قرار دارد.")
except ET.ParseError:
    print(f"خطا: فایل '{file_name}' یک فایل XML معتبر نیست.")