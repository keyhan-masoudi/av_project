import os
from config import Config

def split_xml_file(input_file, output_dir, chunk_size=100):
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Splitting {input_file} into {output_dir}...")
    
    current_chunk = -1
    out_f = None
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            
            if stripped.startswith('<timestep'):
                parts = stripped.split('time="')
                if len(parts) > 1:
                    t = float(parts[1].split('"')[0])
                    chunk_idx = int(t) // chunk_size
                    
                    if chunk_idx != current_chunk:
                        if out_f is not None:
                            out_f.write("</data>\n")
                            out_f.close()
                            
                        current_chunk = chunk_idx
                        chunk_path = os.path.join(output_dir, f"chunk_{current_chunk}.xml")
                        out_f = open(chunk_path, 'w', encoding='utf-8')
                        out_f.write('<?xml version="1.0" encoding="UTF-8"?>\n<data>\n')
                        
            if out_f is not None and not stripped.startswith('<?xml') and not stripped.startswith('<data>') and not stripped.startswith('</data>'):
                out_f.write(line)
                
    if out_f is not None:
        out_f.write("</data>\n")
        out_f.close()
        
    print(f"Finished splitting {input_file}.")

if __name__ == "__main__":
    c_size = Config.CHUNK_SIZE
    
    split_xml_file("D:/av_project/data/vehicles_raw.xml", "D:/av_project/data/vehicles", c_size)
    split_xml_file("D:/av_project/data/tasks_raw.xml", "D:/av_project/data/tasks", c_size)
    split_xml_file("D:/av_project/data/hard_tasks_raw.xml", "D:/av_project/data/hard_tasks", c_size)