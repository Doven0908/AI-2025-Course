import json
import os

def update_template_field_inplace(file_path, new_template_path):
    """
    直接修改原文件中的template字段
    
    Args:
        file_path: JSONL文件路径
        new_template_path: 新的template路径
    """
    
    updated_count = 0
    temp_file = file_path + ".tmp"
    
    with open(file_path, 'r', encoding='utf-8') as f_in, \
         open(temp_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
                old_template = obj.get('template', '')
                obj['template'] = new_template_path
                
                f_out.write(json.dumps(obj) + '\n')
                print(f"第{line_num}行: {old_template} -> {new_template_path}")
                updated_count += 1
                
            except json.JSONDecodeError as e:
                print(f"错误: 第{line_num}行JSON解析失败 - {e}")
                f_out.write(line + '\n')
    
    # 用临时文件替换原文件
    os.replace(temp_file, file_path)
    print(f"\n完成! 共更新了 {updated_count} 个对象的template字段")

def main():
    # 配置参数
    file_path = "human_eval_GroupChatDebughelper5.jsonl"  # 要修改的文件名
    new_template_path = "/mnt/sdb1/zbc/autogen/HumanEval/Templates/GroupChatDebughelper5"
    
    # 执行更新（直接修改原文件）
    update_template_field_inplace(file_path, new_template_path)

if __name__ == "__main__":
    main()
    