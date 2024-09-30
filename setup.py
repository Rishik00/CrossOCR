## For installing dependencies, if any more are to be added they can be included in the pip_libs 
## or linux_libs list

from typing import List
import subprocess

linux_libs: List[str] = ['tesseract-ocr', 
              'tesseract-ocr-hin', 
              'tesseract-ocr-eng'
            ]

pip_libs: List[str] = ['pytesseract',
            'pillow' ,
            'sentence-transformers', 
            'paddlepaddle',
            'paddleocr',
            'PyPDF2', 
            'pdf2image', 
            'langchain', 
            'langchain_community', 
            'langchain_groq', 
            'faiss-cpu'
        ]

def linux_install(
        libs_list: List[str]
    ):

    for lib in libs_list:    
        print(f'Running for {lib}')
        
        linux_install_command = f'sudo apt -qq install {lib}'
        windows_install_command = f''
        subprocess.run(linux_install_command, shell=True, check=True)

def pip_install(
        libs_list: List[str]
    ):

    for lib in libs_list:
        print(f'Running for {lib}')
        
        pip_command = f'pip install -q {lib}'
        subprocess.run(pip_command, shell=True, check=True)

if __name__ == "__main__":
    linux_install(linux_libs)
    pip_install(pip_libs)