#######################
import re
from . import text_cleaning as clean


def _text_splitting(df):
    """
    :param df:
    :return:
    """
    df = df[(df.performed_action_text.str.lower().str.contains('just')) | (
        df.performed_action_text.str.lower().str.contains('comment'))]
    df = df.sort_values(['date_performed'], ascending=False).drop_duplicates(
        subset=['customer_id', 'task_id', 'outcome'], keep='first')
    
    print("%d records after dropping duplicates" % df.shape[0])
    
    df['raw_comments'] = df['performed_action_text'].apply(
        lambda x: re.split('justification:|justificationn|justification|justifcation|justfication|comments',
                           x.lower())[-1]
        if (x.lower().find('just') != -1) | (x.lower().find('comment') != -1) else x)

    df['UI_data'] = df['performed_action_text'].apply(
        lambda x: re.split('justification:|justificationn|justification|justifcation|justfication|comments',
                           x.lower())[0]
        if (x.lower().find('justif') != -1) | (x.lower().find('comment') != -1) else x)

    df['annotations'] = df['raw_comments'].apply(
        lambda x: re.split('\*\*\*ari\*\*\*|\*\*\*ari_cbcc_reaudit\*\*\*|\*\*\*ari rcw\*\*\*',
                           x.lower())[0])
    df['tags'] = df['raw_comments'].apply(
        lambda x: re.split('##|@@', x.lower())[1] if x.lower().find('##') != -1 else 'not available')
    df['tags'] = df['tags'].apply(lambda x: x if len(x) < 50 else 'not available')
    df['blurb'] = df['raw_comments'].apply(
        lambda x: re.split('@@', x.lower())[1] if x.lower().find('@@') != -1 else 'not available')

    return df


def _text_cleaning(df):
    """
    :param df:
    :return:
    """
    # print('----------start processing------------')
    df['annotations'] = [clean._remove_empty_anno(sentence) for sentence in df.annotations]
    df = df[~(df['annotations'].isna() & df['annotations'] == 'No comments')].reset_index(drop=True)
    # print('----------empty annotation removal done!------------')

    if 'clean_comments' in list(df):
        df = df.drop(columns='clean_comments')
    df['clean_comments'] = [clean._decontracted(sentence) for sentence in df.annotations]
    # print('----------annotation lemmatization done!------------')
    df['clean_comments'] = [clean._replace_orderid(sentence) for sentence in df.clean_comments]
    # print('----------order id removal done!------------')
    df['clean_comments'] = clean.annotation_process(df['clean_comments'])
    # print('----------annotation regularization done!------------')
    df['clean_comments'] = [clean._replace_date(sentence) for sentence in df.clean_comments]
    # print('----------date removal done!------------')
    df['clean_comments'] = [clean._replace_days(sentence) for sentence in df.clean_comments]
    # print('----------Procession job complete!!!------------')
    df['clean_comments'] = [clean._remove_number(sentence) for sentence in df.clean_comments]
    # print('----------numbers removal done!------------')

    return df


def _text_cleaning_for_prompt_input(df):
    """
    :param df:
    :return:
    """
    # print('----------start processing------------')
    df['annotations'] = [clean._remove_empty_anno(sentence) for sentence in df.annotations]
    df = df[~(df['annotations'].isna() & df['annotations'] == 'No comments')].reset_index(drop=True)
    # print('----------empty annotation removal done!------------')

    if 'clean_comments_for_prompt' in list(df):
        df = df.drop(columns='clean_comments_for_prompt')
    df['clean_comments_for_prompt'] = [clean._decontracted(sentence) for sentence in df.annotations]
    # print('----------annotation lemmatization done!------------')
    # df['clean_comments_for_prompt'] = [clean._replace_orderid(sentence) for sentence in df.clean_comments]
    # print('----------order id removal done!------------')
    df['clean_comments_for_prompt'] = clean.annotation_process(df['clean_comments_for_prompt'])
    # print('----------annotation regularization done!------------')
    # df['clean_comments_for_prompt'] = [clean._replace_date(sentence) for sentence in df.clean_comments]
    # print('----------date removal done!------------')
    # df['clean_comments_for_prompt'] = [clean._replace_days(sentence) for sentence in df.clean_comments]
    # print('----------Procession job complete!!!------------')
    # df['clean_comments'] = [clean._remove_number(sentence) for sentence in df.clean_comments]
    # print('----------numbers removal done!------------')

    return df


def _tag_cleanning(df):
    """
    :param df:
    :return:
    """
    if 'clean_tags' in list(df):
        df = df.drop(columns='clean_tags')
    df['clean_tags'] = [clean._remove_number(tag) for tag in df['tags']]
    df['clean_tags'] = [clean._remove_punctuation(tag) for tag in df['clean_tags']]
    df['clean_tags'] = [clean._remove_whitespace(tag) for tag in df['clean_tags']]

    if 'clean_blurb' in list(df):
        df = df.drop(columns='clean_blurb')
    df['clean_blurb'] = [clean._remove_number(tag) for tag in df['blurb']]
    df['clean_blurb'] = [clean._remove_punctuation(tag) for tag in df['clean_blurb']]
    df['clean_blurb'] = [clean._remove_whitespace(tag) for tag in df['clean_blurb']]
    return df


def goldminer_processor(df):
    """
    :param df:
    :return:
    """
    df = _text_splitting(df)
    df = _text_cleaning(df)
    df = _tag_cleanning(df)

    return df


def goldminer_processor_for_prompt_input(df):
    """
    :param df:
    :return:
    """
    df = _text_splitting(df)
    df = _text_cleaning_for_prompt_input(df)
    df = _tag_cleanning(df)

    return df


def process_summary_string(df):
    """
    Process a dataframe column containing SummaryRC strings and extract key-value pairs into new columns.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    
    Returns:
    pandas.DataFrame: Original dataframe with additional columns from extracted key-value pairs
    """
    
    def extract_pairs(input_string):
        # Dictionary to store results for this row
        row_dict = {}
        
        try:
            # Extract the content within square brackets
            account_relation = re.findall(r'\[(.*?)\]', input_string)[0]
            row_dict['account_relation'] = account_relation
            
            # Remove the square bracket part from the string
            cleaned_string = re.sub(r'summaryrc\[.*?\];\s*', '', input_string)
            
            # Split the string by semicolon and create key-value pairs
            pairs = [pair.strip() for pair in cleaned_string.split(';')]
            
            # Process each pair
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    # Clean the key: replace spaces with underscore and remove special characters
                    key = key.strip().replace(' ', '_').lower()
                    value = value.strip()
                    row_dict[key] = value
                    
        except Exception as e:
            print(f"Error processing string: {e}")
            
        return row_dict
    
    # Apply the extraction function to each row and create a list of dictionaries
    extracted_data = df["annotations"].apply(extract_pairs)
    
    # Convert the series of dictionaries to a dataframe
    extracted_df = pd.DataFrame(extracted_data.tolist())
    
    # Combine the original dataframe with the new columns
    # First, drop any columns in the original df that have the same names as our new columns
    overlap_columns = set(df.columns).intersection(set(extracted_df.columns))
    if overlap_columns:
        print(f"Warning: The following columns already exist and will be overwritten: {overlap_columns}")
    
    result_df = pd.concat([df, extracted_df], axis=1)
    
    return result_df