from flask import Flask, jsonify, render_template, request, send_file
import subprocess
import os

app = Flask(__name__,
            template_folder='templates3',  
            static_folder='static3')  


TEXT_FILE_PATH = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/datas/all_text.text"
CODEBOOK_INDICES_PATH = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/datas/all_codebook_indices-ccc.txt"
REFERENCE_FILE_PATH = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/datas/all_whole_body_downsampled_data-ccc.txt"
OUTPUT_CODEBOOK_PATH = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/单步生成对比-test_codebook_indices_generated.txt"
OUTPUT_REFERENCE_PATH = "/mnt/sda/caochengcheng/一条龙/演示/4-可视化/单步真实值-test_whole_body_downsampled_data.txt"
VISUALIZATION_SCRIPT = "/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/自回归生成推理-单步生成对比.py"


GIF_PATH_1 = "/mnt/sda/caochengcheng/一条龙/演示/自回归生成阶段输出可视化/单步生成视频/_0.gif"
GIF_PATH_2 = "/mnt/sda/caochengcheng/一条龙/演示/自回归生成阶段输出可视化/单步生成视频/_0_comparison.gif"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


def check_text_in_file(input_text):
   
    try:
        with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if input_text.strip() == line.strip():
                    return i  
        return -1  
    except Exception as e:
        print(f"检查文本时出错: {str(e)}")
        return -1


def process_existing_text(line_number):
    """处理已存在的文本，复制对应的codebook indices和参照数据"""
    try:
      
        with open(CODEBOOK_INDICES_PATH, 'r', encoding='utf-8') as file:
            codebook_lines = file.readlines()
            if line_number < len(codebook_lines):
                codebook_content = codebook_lines[line_number]

          
                with open(OUTPUT_CODEBOOK_PATH, 'w', encoding='utf-8') as output_file:
                    output_file.write(codebook_content)


        with open(REFERENCE_FILE_PATH, 'r', encoding='utf-8') as file:
            reference_lines = file.readlines()
            if line_number < len(reference_lines):
                reference_content = reference_lines[line_number]

               
                with open(OUTPUT_REFERENCE_PATH, 'w', encoding='utf-8') as output_file:
                    output_file.write(reference_content)

       
        result = subprocess.run(['python', VISUALIZATION_SCRIPT],
                                capture_output=True, text=True, check=True)

        return True, result.stdout

    except Exception as e:
        print(f"处理已存在文本时出错: {str(e)}")
        return False, str(e)


@app.route('/run-script', methods=['POST'])
def run_script():
    try:

        data = request.get_json()
        input_text = data.get('text', '')

        if not input_text:
            return jsonify(message='请提供文本内容'), 400

      
        line_number = check_text_in_file(input_text)

        if line_number >= 0:
          
            success, output = process_existing_text(line_number)
            if success:
                return jsonify(
                    message='该动作有真实值作为对照，生成视频如右下方所示（左下方视频为上一次生成结果）',
                    output=output,
                    found_in_file=True,
                    show_gifs=True  
                )
            else:
                return jsonify(
                    message='处理文本时出错',
                    error=output,
                    found_in_file=True,
                    show_gifs=False
                ), 500
        else:
          
            result = subprocess.run([
                'python',
                '/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/自回归生成推理-单步生成.py',
                '--text', input_text
            ], capture_output=True, text=True, check=True)

   
            return jsonify(
                message='该动作没有真实值作为对照，生成视频如左下方所示（右下方视频为上一次生成结果）',
                output=result.stdout,
                found_in_file=False,
                show_gifs=True  # 添加标志以显示GIF
            )

    except subprocess.CalledProcessError as e:
        return jsonify(message='脚本运行失败: ' + str(e), error=e.stderr, show_gifs=False), 500
    except Exception as e:
        return jsonify(message='发生错误: ' + str(e), show_gifs=False), 500


@app.route('/run-full-script', methods=['POST'])
def run_full_script():
    try:
       
        result = subprocess.run([
            'python',
            '/mnt/sda/caochengcheng/一条龙/演示/1-VQVAE2014T/自回归生成推理-全部生成.py'
        ], capture_output=True, text=True, check=True)

        
        return jsonify(message='完整生成脚本成功运行，生成完成', output=result.stdout, show_gifs=True)

    except subprocess.CalledProcessError as e:
        return jsonify(message='完整生成脚本运行失败: ' + str(e), error=e.stderr, show_gifs=False), 500
    except Exception as e:
        return jsonify(message='发生错误: ' + str(e), show_gifs=False), 500



@app.route('/get-gif/<int:gif_id>', methods=['GET'])
def get_gif(gif_id):
    try:
        if gif_id == 1:
            return send_file(GIF_PATH_1, mimetype='image/gif')
        elif gif_id == 2:
            return send_file(GIF_PATH_2, mimetype='image/gif')
        else:
            return "GIF not found", 404
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002)  
