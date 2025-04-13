import csv
import re
import pandas as pd
from ckiptagger import WS, POS, NER, construct_dictionary, data_utils  # 中文斷詞、詞性標註、命名實體識別
from config import * # 載入設定檔
from collections import Counter # 移到檔案頂部
import os # 移到檔案頂部
from typing import List, Dict, Tuple, Any # Import required types, added Any for read_coldata


def check_and_create_folder(folder_path):
    """
    檢查指定路徑的資料夾是否存在，若不存在則建立。

    參數:
    folder_path (str): 要檢查或建立的資料夾路徑。
    """
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' not found. Creating folder...")
        os.makedirs(folder_path, exist_ok=True) # exist_ok=True 避免資料夾已存在時引發錯誤
        print(f"Folder '{folder_path}' created successfully.")
    # else: # 可選：如果資料夾已存在，可以印出訊息
    #     print(f"Folder '{folder_path}' already exists.")


def download_ckip_data():
    """
    檢查 CKIP 模型資料目錄是否存在且非空，若不存在或為空則下載 CKIP 模型資料。
    此函數會下載約 2GB 的資料至 ./data.zip，並解壓縮至設定檔中 ckip_model_data_path 指定的目錄。
    """
    # 檢查並建立 config.py 中定義的 ckip_model_data_path
    check_and_create_folder(ckip_model_data_path) # Use new variable name

    # 檢查模型資料夾路徑是否存在，或該路徑下是否為空目錄
    if not os.path.exists(ckip_model_data_path) or len(os.listdir(ckip_model_data_path)) == 0: # Use new variable name
        # Use new variable name in print statement
        print(f"CKIP model data not found or folder is empty in '{ckip_model_data_path}'. Downloading data...")
        # 從 Google Drive 下載 CKIP 模型資料
        # 檔案大小約 2GB，將下載至 ./data.zip 並解壓縮至 ckip_model_data_path 指定的目錄
        # data_utils.download_data_url("./")  # 備用下載來源：從 IIS 伺服器下載
        data_utils.download_data_gdown("./")  # 從 Google Drive 下載
        print("CKIP model data downloaded and extracted.")
    else:
        # Use new variable name in print statement
        print(f"CKIP model data found in '{ckip_model_data_path}'. Skipping download.")

#-------------------------------------------------------------------------------

def load_removes(file_path: str) -> List[str]: # Use List
    """
    從指定檔案載入需要被移除的字詞清單 (停用詞)。

    參數:
    file_path (str): 包含需要移除字詞的檔案路徑 (通常是 CSV 格式)。

    回傳:
    List[str]: 需要被移除的字詞清單。
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

def load_synonyms(synonym_file: str) -> List[List[str]]: # Use List
    """
    從指定的 CSV 文件載入同義詞詞典，並處理資料格式。
    CSV 檔案中，每一欄代表一組同義詞。

    參數:
    synonym_file (str): 同義詞 CSV 文件的路徑。

    回傳:
    List[List[str]]: 處理後的同義詞列表，每個子列表包含一組同義詞。
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


def read_coldata(df: pd.DataFrame, col_no: int) -> List[Any]: # Use List[Any] for mixed types
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

def initialize_ckip_models(use_cuda=False) -> WS: # Update return type hint
    """
    初始化 CKIP Tagger 模型 (WS). POS and NER are no longer initialized here.

    參數:
    use_cuda (bool): 是否嘗試使用 GPU 加速。預設為 False。

    回傳:
    WS: CKIP Word Segmentation 模型實例。
    """
    print("Initializing CKIP WS model...")
    # Use new variable name
    ws = WS(ckip_model_data_path, disable_cuda=not use_cuda) # Word Segmentation (斷詞)
    # pos = POS(ckip_model_data_path, disable_cuda=not use_cuda) # Part-of-speech tagging (詞性標註) - Removed
    # ner = NER(ckip_model_data_path, disable_cuda=not use_cuda) # Named-entity recognition (命名實體識別) - Removed
    print("CKIP WS model initialized.")
    return ws # Return only the ws instance

def preprocess_text_list(data: List) -> List[str]: # Use List
    """
    對文本列表進行預處理，移除空白字元。

    參數:
    data (List): 包含待處理文本字串的列表。

    回傳:
    List[str]: 處理後的文本列表。
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

def take_wordCounts(ws: WS, sentence_list: List[str], synonymes: List[List[str]]) -> Dict[str, int]: # Use List, Dict
    """
    使用已初始化的 CKIP WS 模型對預處理過的句子列表進行斷詞並統計詞頻。
    POS and NER parameters removed.

    參數:
    ws (WS): 已初始化的 CKIP Word Segmentation 模型。
    sentence_list (List[str]): 預處理過的句子列表。
    synonymes (List[List[str]]): 從同義詞檔案載入的同義詞列表。

    回傳:
    Dict[str, int]: 詞頻統計結果，鍵為詞彙，值為該詞彙出現的次數，按次數降序排列。
    """
    # 載入同義詞詞典並建立權重
    word_to_weight = {string: 2 for string_list in synonymes for string in string_list if len(string) > 2}
    dictionary = construct_dictionary(word_to_weight)

    # 進行斷詞
    print("Performing word segmentation...")
    word_sentence_list = ws(sentence_list, coerce_dictionary=dictionary)
    # POS/NER calls removed

    # 統計斷詞結果的詞頻
    print("Counting words...")
    words = []
    for word_list in word_sentence_list:
        words.extend(word_list)
    word_counts = dict(Counter(words).most_common())
    print("Word counting finished.")
    return word_counts

def calculate_removed_words(counts: Dict[str, int], removes_list: List[str]) -> Dict[str, int]: # Use Dict, List
    """
    Identifies words from the counts dictionary that are present in the removes_list.
    Operates on a copy and does not modify the original counts dictionary.

    參數:
    counts (Dict[str, int]): 原始詞頻統計結果字典。
    removes_list (List[str]): 從停用詞檔案載入的停用詞列表。

    回傳:
    Dict[str, int]: A dictionary containing the removed words and their counts.
    """
    counts_copy = counts.copy() # Create a copy to avoid modifying the original
    removeds = {} # 初始化儲存被移除詞彙及其次數的字典

    # 迭代停用詞列表
    for rmw in removes_list: # Use the parameter removes_list
        if rmw in counts_copy:
            # Get the count from the copy and store it
            no = counts_copy.pop(rmw) # Pop from copy to get count and mark as processed here
            removeds[rmw] = no
        # No need for try-except if using 'in' check

    return removeds

def write_removed_files(removeds: dict, col_no: int, output_dir: str):
    """
    Writes the removed words and their counts to output files.

    參數:
    removeds (dict): Dictionary of removed words and counts.
    col_no (int): Column number for naming files.
    output_dir (str): Directory to save the files.
    """
    # 建立輸出排除詞檔案的路徑
    removeout_path = os.path.join(output_dir, f'排除詞_{col_no}.txt')
    with open(removeout_path, 'w', encoding='utf-8') as f:
        for key, value in removeds.items():
            f.write(f'{key}:{value}\n')

    # 建立輸出排除詞文字雲檔案的路徑
    removewdcnt_path = os.path.join(output_dir, f'排除詞文字雲_{col_no}.txt')
    with open(removewdcnt_path, 'w', encoding='utf-8') as f:
        for key, value in removeds.items():
            for _ in range(value): # Use _ for unused loop variable
                f.write(f'{key} ')

def calculate_synonym_groups(counts: Dict[str, int], synonyms: List[List[str]]) -> Tuple[Dict[str, int], Dict[str, int]]: # Use Dict, List, Tuple
    """
    Aggregates counts for synonym groups and identifies words removed during aggregation.
    Operates on a copy and does not modify the original counts dictionary.

    參數:
    counts (Dict[str, int]): 原始詞頻統計結果字典。
    synonyms (List[List[str]]): 從同義詞檔案載入的同義詞列表。

    回傳:
    Tuple[Dict[str, int], Dict[str, int]]:
        - synonym_aggregates (dict): Aggregated counts with the first synonym as the key.
        - processed_synonyms (dict): All individual synonyms processed with their original counts.
    """
    counts_copy = counts.copy() # Operate on a copy
    synonym_aggregates = {} # Stores aggregated counts {representative_word: total_count}
    processed_synonyms = {} # Stores individual synonyms processed {synonym: original_count}

    for sw in synonyms:
        totalw = 0
        if not sw:
            continue
        kwywd = sw[0]
        group_synonyms_found = {}

        for w in sw:
            if w in counts_copy:
                count = counts_copy.pop(w) # Pop from copy to aggregate and mark as processed
                totalw += count
                group_synonyms_found[w] = count
                processed_synonyms[w] = count # Track individual processed synonyms

        if totalw != 0:
            synonym_aggregates[kwywd] = totalw

    return synonym_aggregates, processed_synonyms


def write_synonym_files(synonym_aggregates: dict, col_no: int, output_dir: str):
    """
    Writes the aggregated synonym counts to output files.

    參數:
    synonym_aggregates (dict): Dictionary of aggregated synonym counts.
    col_no (int): Column number for naming files.
    output_dir (str): Directory to save the files.
    """
    # 建立輸出同義詞檔案的路徑
    synonymout_path = os.path.join(output_dir, f'同義詞_{col_no}.txt')
    with open(synonymout_path, 'w', encoding='utf-8') as f:
        for key, value in synonym_aggregates.items():
            f.write(f'{key}:{value}\n')

    # 建立輸出同義詞文字雲檔案的路徑
    synonymws_path = os.path.join(output_dir, f'同義詞文字雲_{col_no}.txt')
    with open(synonymws_path, 'w', encoding='utf-8') as f:
        for key, value in synonym_aggregates.items():
            for _ in range(value): # Use _ for unused loop variable
                f.write(f'{key} ')


def calculate_filtered_counts(word_counts: Dict[str, int], remove_list: List[str], synonym_list: List[List[str]]) -> Dict[str, int]: # Use Dict, List
    """
    Removes stopwords and all synonyms from the word counts.

    參數:
    word_counts (Dict[str, int]): 詞頻統計結果字典。
    remove_list (List[str]): 停用詞列表。
    synonym_list (List[List[str]]): 同義詞列表。

    回傳:
    Dict[str, int]: 過濾後的詞頻統計結果字典。
    """
    filtered_counts = word_counts.copy()
    words_to_exclude = set(remove_list)
    for group in synonym_list:
        for word in group:
            if word:
                words_to_exclude.add(word)

    for word in words_to_exclude:
        if word in filtered_counts:
            filtered_counts.pop(word)

    return filtered_counts

def write_filtered_file(filtered_counts: dict, col_no: int, output_dir: str):
    """
    Writes the final filtered word counts to a file.

    參數:
    filtered_counts (dict): The filtered word counts.
    col_no (int): Column number for naming the file.
    output_dir (str): Directory to save the file.
    """
    filtered_output_path = os.path.join(output_dir, f'過濾詞_{col_no}.txt')
    with open(filtered_output_path, 'w', encoding='utf-8') as f:
        for key, value in filtered_counts.items():
            f.write(f'{key}:{value}\n')


def process_multiple_columns(column_numbers, use_cuda=False):
    """
    Process multiple columns from an Excel file using CKIP text analysis pipeline.
    Refactored to separate calculation and file writing.
    """
    # Download CKIP model data if needed
    download_ckip_data()

    # Initialize models (with or without CUDA) - Now correctly receives only ws_model
    ws_model: WS = initialize_ckip_models(use_cuda=use_cuda) # Assign the returned WS instance

    # Read the Excel file
    print(f"Reading data from: {data_file}")
    df = pd.read_excel(data_file)

    # Load synonyms and stopwords (only need to do this once)
    print(f"Loading synonyms from: {synonym_file}")
    synonym_list = load_synonyms(synonym_file)

    print(f"Loading stopwords from: {remove_file}")
    remove_list = load_removes(remove_file)

    # Ensure output directory exists (moved here for clarity)
    check_and_create_folder(word_cloud_path)

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
        # Pass only ws_model, pos/ner removed
        word_counts = take_wordCounts(ws_model, processed_text, synonym_list)

        # Calculate and write removed words (stopwords)
        print("Processing removals...")
        removed_words_dict = calculate_removed_words(word_counts, remove_list)
        write_removed_files(removed_words_dict, col, word_cloud_path)

        # Calculate and write synonym aggregations
        print("Processing synonyms...")
        # Note: calculate_synonym_groups now returns two dicts, we only need the first for writing synonym files
        synonym_aggregates_dict, _ = calculate_synonym_groups(word_counts, synonym_list)
        write_synonym_files(synonym_aggregates_dict, col, word_cloud_path)

        # Calculate and write final filtered word counts
        print("Filtering word counts...")
        filtered_counts_dict = calculate_filtered_counts(word_counts, remove_list, synonym_list)
        write_filtered_file(filtered_counts_dict, col, word_cloud_path)

        print(f"Column {col} processing complete.")

    print(f"\nAll columns processing complete. Output files generated in: {word_cloud_path}")


# Example usage in main
if __name__ == '__main__':
    # Process multiple columns (example)
    columns_to_process = [23]
    process_multiple_columns(columns_to_process, use_cuda=False)
