import csv
import re
import pandas as pd
from ckiptagger import WS, POS, NER, construct_dictionary, data_utils  # 中文斷詞、詞性標註、命名實體識別
from config import * # 載入設定檔
from collections import Counter # 移到檔案頂部
import os # 移到檔案頂部

def download_ckip_data():
    """
    檢查 CKIP 模型資料目錄是否存在且非空，若不存在或為空則下載 CKIP 模型資料。
    此函數會下載約 2GB 的資料至 ./data.zip，並解壓縮至設定檔中 data_folder_path 指定的目錄。
    """
    # 檢查模型資料夾路徑是否存在，或該路徑下是否為空目錄
    if not os.path.exists(data_folder_path) or len(os.listdir(data_folder_path)) == 0:
        # 從 Google Drive 下載 CKIP 模型資料
        # 檔案大小約 2GB，將下載至 ./data.zip 並解壓縮至 data_folder_path 指定的目錄
        # data_utils.download_data_url("./")  # 備用下載來源：從 IIS 伺服器下載
        data_utils.download_data_gdown("./")  # 從 Google Drive 下載

#-------------------------------------------------------------------------------

def load_removes(file_path: str) -> list:
    """
    從指定檔案載入需要被移除的字詞清單 (停用詞)。

    參數:
    file_path (str): 包含需要移除字詞的檔案路徑 (通常是 CSV 格式)。

    回傳:
    list: 需要被移除的字詞清單。
    """
    removes = []  # 初始化儲存停用詞的列表
    # 開啟指定的檔案，使用 utf-8-sig 編碼處理可能存在的 BOM，並指定換行符號
    with open(file_path, encoding='utf-8-sig', newline='\r\n') as csvfile:
        spamreader = csv.reader(csvfile) # 建立 CSV 讀取器
        for row in spamreader: # 逐行讀取
            # 使用列表生成式移除列表中的空字串，確保不加入空字串到停用詞列表
            result = [word for word in row if word != ""]
            removes += result # 將該行的有效字詞加入停用詞列表
    #print(f"\n停用詞列表 \n{removes}") # 可選：印出載入的停用詞列表以供檢查
    return removes # 回傳停用詞列表

#-------------------------------------------------------------------------------

def load_synonyms(synonym_file: str) -> list:
    """
    從指定的 CSV 文件載入同義詞詞典，並處理資料格式。
    CSV 檔案中，每一欄代表一組同義詞。

    參數:
    synonym_file (str): 同義詞 CSV 文件的路徑。

    回傳:
    list: 處理後的同義詞列表，每個子列表包含一組同義詞。
          例如: [['電腦', '計算機'], ['手機', '行動電話']]
    """
    # 讀取同義詞 CSV 文件，使用 pandas 讀取
    # .T 進行轉置，讓原本的欄變成列
    # reset_index() 重設索引，方便後續迭代
    synonym_df = pd.read_csv(synonym_file).T.reset_index()

    # 將 DataFrame 的每一列轉換為列表，並移除 NaN (空值)
    # 每個子列表代表一組同義詞
    synonymes = [row.dropna().tolist() for _, row in synonym_df.iterrows()]

    # 處理同義詞中的空格，確保詞彙匹配的準確性
    result = [] # 初始化處理結果列表
    for string_list in synonymes: # 迭代每一組同義詞列表
        new_list = [] # 初始化處理單組同義詞的列表
        for string in string_list: # 迭代該組同義詞中的每個詞
            # 移除每個詞彙內部可能存在的空格
            new_list.append(string.replace(' ', ''))
        result.append(new_list) # 將處理完畢的單組同義詞列表加入最終結果

    # 更新同義詞列表為處理空格後的結果
    synonymes = result
    #print(f"\n同義詞詞典 \n{synonymes}") # 可選：印出載入的同義詞詞典以供檢查
    return synonymes # 回傳處理後的同義詞列表


def read_coldata(df: pd.DataFrame, col_no: int) -> list: # Corrected return type hint
    """
    從 pandas DataFrame 中讀取指定欄位編號前綴的資料。

    參數:
    df (pd.DataFrame): 輸入的 DataFrame。
    col_no (int): 欲搜尋的欄位編號前綴 (例如：若欄位名為 "23. 問題描述"，則輸入 23)。

    回傳:
    list: 包含符合指定欄位編號前綴的欄位資料 (已移除 NaN 值)。

    範例:
    如果 DataFrame 'df' 的欄位為 '1. Name', '2. Age', '3. Address'，
    呼叫 read_coldata(df, 2) 將回傳 '2. Age' 欄位的資料列表 (移除 NaN 後)。

    注意:
    - 欄位編號前綴會被轉換為字串，並附加一個句點以匹配欄位名稱中的確切模式。
    """

    # 將欄位編號轉換為字串並附加句點，以建立搜尋模式 (例如 "23.")
    search_pattern = str(col_no) + "."

    # 使用列表生成式找出 DataFrame 中所有以 search_pattern 開頭的欄位名稱
    col = [column for column in df.columns if column.startswith(search_pattern)]

    # 選取第一個符合模式的欄位資料 (假設每個編號前綴只對應一個欄位)
    # .dropna() 移除該欄位中的空值 (NaN)
    col_data = df[col[0]].dropna()

    # 將選取的欄位資料轉換為 Python list 並回傳
    return col_data.to_list()

def initialize_ckip_models(use_cuda=False):
    """
    初始化 CKIP Tagger 模型 (WS, POS, NER)。

    參數:
    use_cuda (bool): 是否嘗試使用 GPU 加速。預設為 False。

    回傳:
    tuple: 包含 WS, POS, NER 模型實例的元組。
    """
    print("Initializing CKIP models...")
    ws = WS(data_folder_path, disable_cuda=not use_cuda) # Word Segmentation (斷詞)
    pos = POS(data_folder_path, disable_cuda=not use_cuda) # Part-of-speech tagging (詞性標註)
    ner = NER(data_folder_path, disable_cuda=not use_cuda) # Named-entity recognition (命名實體識別)
    print("CKIP models initialized.")
    return ws, pos, ner

def preprocess_text_list(data: list) -> list:
    """
    對文本列表進行預處理，移除空白字元。

    參數:
    data (list): 包含待處理文本字串的列表。

    回傳:
    list: 處理後的文本列表。
    """
    text = [] # 初始化儲存預處理後文本的列表
    for row in data: # 迭代輸入的文本列表
        if isinstance(row, str): # 檢查是否為字串型別 (更 Pythonic 的方式)
            # 使用正規表達式移除字串中所有空白字元
            res, _ = re.subn('\s+', '', row) # _ 忽略替換次數
            text.append(res) # 將處理後的字串加入列表
        # 可以選擇性地處理非字串的情況，例如轉換或忽略
        # else:
        #     print(f"Warning: Skipping non-string data: {row}")
    return text

def take_wordCounts(ws: WS, pos: POS, ner: NER, sentence_list: list, synonymes: list) -> dict:
    """
    使用已初始化的 CKIP 模型對預處理過的句子列表進行斷詞並統計詞頻。

    參數:
    ws (WS): 已初始化的 CKIP Word Segmentation 模型。
    pos (POS): 已初始化的 CKIP Part-of-speech tagging 模型。
    ner (NER): 已初始化的 CKIP Named-entity recognition 模型。
    sentence_list (list): 預處理過的句子列表。
    synonymes (list): 從同義詞檔案載入的同義詞列表。

    回傳:
    dict: 詞頻統計結果，鍵為詞彙，值為該詞彙出現的次數，按次數降序排列。
    """
    # 載入同義詞詞典並建立權重
    word_to_weight = {string: 2 for string_list in synonymes for string in string_list if len(string) > 2}
    dictionary = construct_dictionary(word_to_weight)

    # 進行斷詞、詞性標註、命名實體識別
    print("Performing word segmentation...")
    word_sentence_list = ws(sentence_list, coerce_dictionary=dictionary)
    #print("Performing POS tagging...")
    #pos_sentence_list = pos(word_sentence_list)
    #print("Performing NER...")
    #entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    # 注意：entity_sentence_list 在此函數中目前未使用，但保留以備將來擴展

    # 統計斷詞結果的詞頻
    print("Counting words...")
    words = []
    for word_list in word_sentence_list:
        words.extend(word_list)
    word_counts = dict(Counter(words).most_common())
    print("Word counting finished.")
    return word_counts

def make_removedFile(counts: dict, col_no: int, removes_list: list):
    """
    根據提供的停用詞列表從詞頻統計結果中移除指定的詞彙，
    並將被移除的詞彙及其次數輸出到指定的檔案。
    此函數操作於輸入字典的副本上，不會修改原始字典。

    參數:
    counts (dict): 原始詞頻統計結果字典。
    col_no (int): 正在處理的原始資料欄位編號，用於命名輸出檔案。
    removes_list (list): 從停用詞檔案載入的停用詞列表。
    """
    counts_copy = counts.copy() # Create a copy to avoid modifying the original
    removeds = {} # 初始化儲存被移除詞彙及其次數的字典

    # 迭代停用詞列表
    for rmw in removes_list: # Use the parameter removes_list
        try:
            # 嘗試從詞頻字典 counts_copy 中移除停用詞 rmw
            # dict.pop(key) 會移除鍵 key 並返回其對應的值
            no = counts_copy.pop(rmw) # Operate on the copy
            # 將被移除的詞彙及其次數存入 removeds 字典
            removeds[rmw] = no
        except KeyError: # 若停用詞原本就不在 counts_copy 中，會引發 KeyError
            # 忽略這個錯誤，繼續處理下一個停用詞
            ...

    # 建立輸出排除詞檔案的路徑
    # word_cloud_path 從 config.py 讀取，是存放文字雲相關檔案的目錄
    removeout_path = word_cloud_path + f'/排除詞_{col_no}.txt'
    # 開啟檔案準備寫入，使用 utf-8 編碼
    with open(removeout_path, 'w', encoding='utf-8') as f:
        # 迭代被移除詞彙字典
        for key, value in removeds.items():
            # 將每個詞彙和其次數寫入檔案，格式為 "詞彙:次數"
            f.write(f'{key}:{value}\n')

    # 建立輸出排除詞文字雲檔案的路徑 (用於生成文字雲的原始文本)
    removewdcnt_path = word_cloud_path + f'/排除詞文字雲_{col_no}.txt'
    # 開啟檔案準備寫入，使用 utf-8 編碼
    with open(removewdcnt_path, 'w', encoding='utf-8') as f:
        # 迭代被移除詞彙字典
        for key, value in removeds.items():
            # 將每個詞彙重複寫入 value (次數) 次，並以空格分隔
            # 這會產生一個適合直接輸入文字雲生成工具的文本
            for i in range(value):
                f.write(f'{key} ')


def make_synonymFile(counts: dict, col_no: int, synonyms: list):
    """
    根據提供的同義詞列表合併詞頻統計結果中的同義詞,
    將合併後的同義詞 (以第一個詞為代表) 及其總次數輸出到指定的檔案。
    此函數操作於輸入字典的副本上，不會修改原始字典。

    參數:
    counts (dict): 原始詞頻統計結果字典。
    col_no (int): 正在處理的原始資料欄位編號，用於命名輸出檔案。
    synonyms (list): 從同義詞檔案載入的同義詞列表。
    """
    counts_copy = counts.copy() # Create a copy to avoid modifying the original
    synonymws = {} # 初始化儲存合併後同義詞及其總次數的字典

    # 迭代每一組同義詞列表
    for sw in synonyms:
        totalw = 0 # 初始化該組同義詞的總次數
        if not sw:  # 如果同義詞組為空列表，則跳過
            continue

        # 使用該組同義詞列表中的第一個詞作為這組同義詞的代表詞 (關鍵詞)
        kwywd = sw[0]

        # 迭代該組同義詞中的每一個詞彙 w
        for w in sw:
            # 檢查該詞彙 w 是否存在於目前的詞頻字典 counts_copy 中
            if w in counts_copy:
                # 如果存在，將其 次數 加入 totalw
                totalw += counts_copy[w] # Read from the copy
                # 從 counts_copy 字典中移除該詞彙 w，避免後續重複計算或被單獨輸出
                counts_copy.pop(w) # Operate on the copy
            # 此處原有的 try...except pass 結構可以省略，因為 if w in counts_copy 已經處理了 KeyError 的情況

        # 如果該組同義詞的總次數不為 0 (表示至少有一個同義詞出現在原始文本中)
        if totalw != 0:
            # 將代表詞 kwywd 和其總次數 totalw 存入 synonymws 字典
            synonymws[kwywd] = totalw

    # 建立輸出同義詞檔案的路徑
    synonymout_path = word_cloud_path + f'/同義詞_{col_no}.txt'
    # 開啟檔案準備寫入，使用 utf-8 編碼
    with open(synonymout_path, 'w', encoding='utf-8') as f:
        # 迭代合併後的同義詞字典
        for key, value in synonymws.items():
            # 將每個代表詞和其總次數寫入檔案，格式為 "代表詞:總次數"
            f.write(f'{key}:{value}\n')

    # 建立輸出同義詞文字雲檔案的路徑 (用於生成文字雲的原始文本)
    synonymws_path = word_cloud_path + f'/同義詞文字雲_{col_no}.txt'
    # 開啟檔案準備寫入，使用 utf-8 編碼
    with open(synonymws_path, 'w', encoding='utf-8') as f:
        # 迭代合併後的同義詞字典
        for key, value in synonymws.items():
            # 將每個代表詞重複寫入 value (總次數) 次，並以空格分隔
            for i in range(value):
                f.write(f'{key} ')

def filter_word_counts(word_counts: dict, remove_list: list, synonym_list: list, col_no: int) -> dict:
    """
    從詞頻統計結果中移除停用詞和所有同義詞（包括代表詞），並將結果存檔。
    
    參數:
    word_counts (dict): 詞頻統計結果字典。
    remove_list (list): 停用詞列表。
    synonym_list (list): 同義詞列表，每個子列表包含一組同義詞。
    col_no (int): 正在處理的原始資料欄位編號，用於命名輸出檔案。
    output_path (str): 儲存輸出檔案的目錄路徑。
    
    回傳:
    dict: 過濾後的詞頻統計結果字典。
    """
    # 創建一個副本，避免修改原始字典
    filtered_counts = word_counts.copy()
    
    # 收集所有需要排除的詞彙
    words_to_exclude = set(remove_list)
    
    # 添加所有同義詞到排除集合中
    for group in synonym_list:
        for word in group:
            if word:  # 確保不加入空字串
                words_to_exclude.add(word)
    
    # 從詞頻字典中移除需要排除的詞彙
    for word in words_to_exclude:
        if word in filtered_counts:
            filtered_counts.pop(word)
    
    #print(f"Filtered word counts: removed {len(words_to_exclude)} words from dictionary")
    #print(f"Original count: {len(word_counts)}, Filtered count: {len(filtered_counts)}")

    # 建立輸出過濾詞檔案的路徑
    filtered_output_path = os.path.join(word_cloud_path, f'過濾詞_{col_no}.txt')
    # 開啟檔案準備寫入，使用 utf-8 編碼
    with open(filtered_output_path, 'w', encoding='utf-8') as f:
        # 迭代過濾後的詞彙字典
        for key, value in filtered_counts.items():
            # 將每個詞彙和其次數寫入檔案，格式為 "詞彙:次數"
            f.write(f'{key}:{value}\n')
    #print(f"Filtered word counts saved to: {filtered_output_path}")
    
    return filtered_counts

def process_multiple_columns(column_numbers, use_cuda=False):
    """
    Process multiple columns from an Excel file using CKIP text analysis pipeline.
    
    This function handles downloading models, initialization, and processes each
    specified column through the entire text analysis workflow including:
    - Word segmentation
    - Filtering by stopwords
    - Handling synonyms
    - Generating output files
    
    Parameters:
    column_numbers (list): List of column numbers to process
    use_cuda (bool): Whether to use GPU acceleration (default: False)
    
    Returns:
    dict: Dictionary with column numbers as keys and their filtered word counts as values
    """
    # Download CKIP model data if needed
    download_ckip_data()

    # Initialize models (with or without CUDA)
    ws_model, pos_model, ner_model = initialize_ckip_models(use_cuda=use_cuda)

    # Read the Excel file
    print(f"Reading data from: {data_file}")
    df = pd.read_excel(data_file)
    
    # Load synonyms and stopwords (only need to do this once)
    print(f"Loading synonyms from: {synonym_file}")
    synonym_list = load_synonyms(synonym_file)
    
    print(f"Loading stopwords from: {remove_file}")
    remove_list = load_removes(remove_file)
    
    # Process each column
    for col in column_numbers:
        print(f"\n=== Processing column {col} ===")
        
        # Read column data
        print(f"Reading data from column prefix: {col}")
        column_data = read_coldata(df, col)
        
        # Preprocess text
        print("Preprocessing text...")
        processed_text = preprocess_text_list(column_data)
        
        # Perform word segmentation and count frequencies
        print("Counting words...")
        # Call the CKIP model to get word counts
        word_counts = take_wordCounts(ws_model, pos_model, ner_model, processed_text, synonym_list)
        
        # Process stopwords
        print("Processing removals...")
        make_removedFile(word_counts, col, remove_list)
        
        # Process synonyms
        print("Processing synonyms...")
        make_synonymFile(word_counts, col, synonym_list)
        
        # Filter word counts and save results
        print("Filtering word counts...")
        # Call the filter function to remove stopwords and synonyms from word counts
        filter_word_counts(word_counts, remove_list, synonym_list, col)
        
        print(f"Column {col} processing complete.")
    
    print(f"\nAll columns processing complete. Output files generated in: {word_cloud_path}")


# Example usage in main
if __name__ == '__main__':
    # Process multiple columns (example)
    columns_to_process = [23]
    process_multiple_columns(columns_to_process, use_cuda=False)
