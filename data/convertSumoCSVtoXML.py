import pandas as pd
import xml.etree.ElementTree as ET

power_list = [2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8]

df = pd.read_csv('sorted_file20_v2.csv')

# اطمینان از اینکه vehicle_id به عنوان int خوانده شده است
df['vehicle_id'] = df['vehicle_id'].astype(int)

root = ET.Element('data')

# گروه‌بندی داده‌ها بر اساس زمان
grouped = df.groupby('time')

for time, group in grouped:
    # ایجاد عنصر timestep با ویژگی time
    timestep = ET.SubElement(root, 'timestep')
    timestep.set('time', str(time))

    for _, row in group.iterrows():
        vehicle_id = int(row['vehicle_id'])
        # ایجاد عنصر vehicle با ویژگی‌های مربوطه
        vehicle = ET.SubElement(timestep, 'vehicle')
        vehicle.set('id', str(vehicle_id))  # حذف .0 با تبدیل به int
        vehicle.set('x', str(row['x']))
        vehicle.set('y', str(row['y']))
        vehicle.set('angle', str(row['angle']))
        vehicle.set('speed', str(row['speed']))

        # اضافه کردن ویژگی power
        if 1 <= vehicle_id <= len(power_list):
            power = power_list[vehicle_id - 1]
            vehicle.set('power', str(power))
        else:
            vehicle.set('power', "0")

# فرمت‌دهی مناسب XML (pretty-print)
ET.indent(root, space="    ", level=0)

tree = ET.ElementTree(root)

tree.write('output.xml', encoding='utf-8', xml_declaration=True)

print("done!")
