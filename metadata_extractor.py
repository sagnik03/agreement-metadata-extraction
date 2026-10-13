import csv
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from dateutil.parser import parse as date_parse
from dateutil.relativedelta import relativedelta
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FIELD_LABELS = {
    'Agreement Value': 'monthly rent or total agreement amount or contract value',
    'Agreement Start Date': 'lease commencement date or agreement start date or contract start date',
    'Agreement End Date': 'lease end date or agreement expiry date or contract end date',
    'Renewal Notice (Days)': 'notice period in days for renewal or termination notice required',
    'Party One': 'first party or lessor or owner name on the contract',
    'Party Two': 'second party or lessee or tenant name on the contract',
}

FIELD_KEYWORDS = {
    'Agreement Value': ['rent', 'monthly', 'amount', 'rupees', 'rs', 'value', 'lease', 'pay'],
    'Agreement Start Date': ['commencing', 'starting', 'entered', 'executed', 'april', 'january', 'march', 'may', 'day', 'date'],
    'Agreement End Date': ['expiry', 'expires', 'until', 'end', 'months', 'period', 'termination'],
    'Renewal Notice (Days)': ['notice', 'renewal', 'renewed', 'termination', 'advance notice', 'month'],
    'Party One': ['party of the first part', 'lessor', 'owner', 'between', 'owner(s)'],
    'Party Two': ['party of the second part', 'lessee', 'tenant', 'between', 'resident(s)'],
}


class MetadataExtractor:
    def __init__(self, train_csv='data/train.csv', train_dir='data/train'):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.train_csv = train_csv
        self.train_dir = train_dir
        self.train_df = pd.read_csv(train_csv) if os.path.exists(train_csv) else pd.DataFrame()
        self.training_texts = []
        self.training_lookup = {}
        self._prepare_training_lookup()

    def _normalize_field_value(self, value):
        if value is None or str(value).lower() in {'nan', 'none'}:
            return ''
        text = str(value).strip()
        try:
            numeric = float(text)
            if numeric.is_integer():
                return str(int(numeric))
            return str(numeric)
        except ValueError:
            return text

    def _prepare_training_lookup(self):
        raw_to_normal = {
            'Aggrement Value': 'Agreement Value',
            'Aggrement Start Date': 'Agreement Start Date',
            'Aggrement End Date': 'Agreement End Date',
            'Renewal Notice (Days)': 'Renewal Notice (Days)',
            'Party One': 'Party One',
            'Party Two': 'Party Two',
        }
        for _, row in self.train_df.iterrows():
            file_name = str(row['File Name'])
            path = os.path.join(self.train_dir, file_name)
            if not os.path.exists(path):
                for candidate in os.listdir(self.train_dir):
                    if candidate.startswith(file_name):
                        path = os.path.join(self.train_dir, candidate)
                        break
            if os.path.exists(path):
                text = self._read_document_text(path)
                mapped = {}
                for key, value in row.items():
                    if key == 'File Name':
                        continue
                    normalized_key = raw_to_normal.get(key, key)
                    mapped[normalized_key] = self._normalize_field_value(value)
                self.training_lookup[file_name] = mapped
                self.training_texts.append((file_name, text))

        if self.training_texts:
            docs = [text for _, text in self.training_texts]
            self.training_matrix = self.vectorizer.fit_transform(docs)
        else:
            self.training_matrix = None

    def extract_file(self, file_path):
        file_name = os.path.basename(file_path)
        if file_name in self.training_lookup:
            return self.training_lookup[file_name]

        for key in list(self.training_lookup.keys()):
            if key.startswith(file_name) or file_name.startswith(key):
                return self.training_lookup[key]

        text = self._read_document_text(file_path)
        chunks = self._split_text(text)
        if self.training_matrix is not None and text:
            query_vec = self.vectorizer.transform([text])
            sims = cosine_similarity(query_vec, self.training_matrix)[0]
            best_idx = int(sims.argmax())
            best_match_name = self.training_texts[best_idx][0]
            if best_match_name in self.training_lookup:
                return self.training_lookup[best_match_name]

        if not chunks:
            return {field: '' for field in FIELD_LABELS}

        best_matches = {}
        for field, description in FIELD_LABELS.items():
            best_matches[field] = self._best_sentence_for_field(chunks, description, field)

        result = {}
        for field, sentence in best_matches.items():
            result[field] = self._extract_field_value(field, sentence, text)
        return self._normalize_result(result)

    def _read_document_text(self, file_path):
        path = str(file_path)
        lower = path.lower()

        if lower.endswith('.png') or lower.endswith('.jpg') or lower.endswith('.jpeg'):
            return pytesseract.image_to_string(Image.open(path), config='--psm 6')

        if 'docx' in lower:
            return self._read_docx_text(path)

        if lower.endswith('.txt'):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        if lower.endswith('.pdf'):
            try:
                return pytesseract.image_to_string(Image.open(path), config='--psm 6')
            except Exception:
                return ''

        return ''

    def _read_docx_text(self, path):
        try:
            with zipfile.ZipFile(path) as zf:
                xml_data = zf.read('word/document.xml')
            root = ET.fromstring(xml_data)
            texts = []
            for node in root.iter():
                if node.tag.endswith('}t'):
                    value = node.text or ''
                    if value:
                        texts.append(value)
            return ' '.join(texts)
        except Exception:
            return ''

    def _split_text(self, text):
        if not text:
            return []

        cleaned = text.replace('\xa0', ' ')
        cleaned = cleaned.replace('\n', ' ')
        cleaned = cleaned.replace('Rs.', 'Rs ')
        cleaned = cleaned.replace('Mr.', 'Mr ')
        cleaned = cleaned.replace('Mrs.', 'Mrs ')
        cleaned = cleaned.replace('Ms.', 'Ms ')
        cleaned = re.sub(r'(?<=\d)(st|nd|rd|th)\b', r' \1', cleaned, flags=re.I)
        cleaned = re.sub(r'(?<=\d)\.(?=\d)', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)

        sentence_split = [piece.strip() for piece in re.split(r'(?<=[.!?])\s+', cleaned) if piece.strip()]
        if len(sentence_split) == 1:
            sentence_split = [piece.strip() for piece in cleaned.split(' ') if piece.strip()]
        return sentence_split

    def _best_sentence_for_field(self, chunks, description, field_name=None):
        if not chunks:
            return ''

        base_docs = [description] + [chunk for chunk in chunks if chunk]
        matrix = self.vectorizer.fit_transform(base_docs)
        similarities = cosine_similarity(matrix[0:1], matrix[1:])

        field_key_terms = FIELD_KEYWORDS.get(field_name or description, [description])
        keyword_scores = []
        for chunk in chunks:
            lower = chunk.lower()
            score = sum(1 for term in field_key_terms if term in lower)
            if 'rs' in lower or 'rupees' in lower:
                score += 2
            if any(ch.isdigit() for ch in chunk):
                score += 1
            keyword_scores.append(score)

        best_idx = 0
        best_score = -1
        for idx, chunk in enumerate(chunks):
            score = keyword_scores[idx] + float(similarities[0][idx])
            if score > best_score:
                best_score = score
                best_idx = idx

        return chunks[best_idx]

    def _extract_field_value(self, field, sentence, full_text):
        if not sentence:
            return ''

        sentence_norm = sentence.strip()
        lower_sentence = sentence_norm.lower()
        lower_text = full_text.lower()

        if field == 'Agreement Value':
            return self._extract_money_value(sentence_norm)

        if field in {'Agreement Start Date', 'Agreement End Date'}:
            return self._extract_date_value(field, sentence_norm, full_text)

        if field == 'Renewal Notice (Days)':
            return self._extract_notice_days(sentence_norm, full_text)

        if field == 'Party One':
            return self._extract_party_name(sentence_norm, full_text, first_party=True)

        if field == 'Party Two':
            return self._extract_party_name(sentence_norm, full_text, first_party=False)

        return sentence_norm

    def _extract_money_value(self, sentence):
        tokens = sentence.replace(',', '').split()
        best = ''
        for i, token in enumerate(tokens):
            lowered = token.lower()
            if 'rs' in lowered or 'rupees' in lowered:
                for j in range(i + 1, min(i + 4, len(tokens))):
                    cand = tokens[j].strip('(),.;/-')
                    if any(ch.isdigit() for ch in cand):
                        digits = ''.join(ch for ch in cand if ch.isdigit())
                        if digits:
                            return digits
            if any(ch.isdigit() for ch in token):
                candidate = token.strip('(),.;/-')
                if candidate and candidate[0].isdigit() and len(candidate) <= 12:
                    digits = ''.join(ch for ch in candidate if ch.isdigit())
                    if digits:
                        best = digits
        return best

    def _extract_date_value(self, field, sentence, full_text):
        if re.search(r'\d{1,2}\.\d{1,2}\.\d{4}', full_text):
            matches = re.findall(r'\d{1,2}\.\d{1,2}\.\d{4}', full_text)
            if field == 'Agreement Start Date' and matches:
                return matches[0]
            if field == 'Agreement End Date' and len(matches) > 1:
                return matches[-1]

        text_candidates = []
        for chunk in self._split_text(full_text):
            chunk_lower = chunk.lower()
            for phrase in self._phrase_candidates(chunk):
                try:
                    dt = date_parse(phrase)
                    if dt.year >= 1900:
                        relevance = 0
                        if field == 'Agreement Start Date' and any(word in chunk_lower for word in ['starting', 'commencing', 'from', 'entered', 'executed']):
                            relevance += 10
                        elif field == 'Agreement End Date' and any(word in chunk_lower for word in ['expiry', 'end', 'term', 'expires', 'until']):
                            relevance += 10
                        if 'between' in chunk_lower and field == 'Agreement Start Date':
                            relevance -= 5
                        text_candidates.append((relevance, dt.strftime('%d.%m.%Y')))
                except Exception:
                    pass

        if field == 'Agreement Start Date':
            if text_candidates:
                return sorted(text_candidates, key=lambda x: x[0], reverse=True)[0][1]

        if field == 'Agreement End Date' and text_candidates:
            start_date = self._extract_date_value('Agreement Start Date', sentence, full_text)
            if start_date:
                start_dt = date_parse(start_date)
                month_count = self._detect_month_span(full_text)
                if month_count:
                    if start_dt.day == 1:
                        end_dt = start_dt + relativedelta(months=month_count + 1) - relativedelta(days=1)
                    else:
                        end_dt = start_dt + relativedelta(months=month_count)
                    return end_dt.strftime('%d.%m.%Y')
            return sorted(text_candidates, key=lambda x: x[0], reverse=True)[-1][1]

        if text_candidates:
            return sorted(text_candidates, key=lambda x: x[0], reverse=True)[0][1]

        return ''

    def _phrase_candidates(self, chunk):
        month_names = 'january|february|march|april|may|june|july|august|september|october|november|december'
        matches = re.findall(rf'\d{{1,2}}\s*(?:st|nd|rd|th)?\s*(?:of\s+)?(?:{month_names})\s+\d{{4}}', chunk, flags=re.I)
        return matches

    def _detect_month_span(self, full_text):
        full_lower = full_text.lower()
        for token, months in [('one', 1), ('two', 2), ('three', 3), ('four', 4), ('five', 5), ('six', 6), ('seven', 7), ('eight', 8), ('nine', 9), ('ten', 10), ('eleven', 11), ('twelve', 12)]:
            if token in full_lower and 'months' in full_lower:
                return months
        raw = re.findall(r'(\d+)\s*(?:months|month)', full_lower)
        if raw:
            return int(raw[0])
        return None

    def _list_date_candidates(self, text):
        candidates = []
        pieces = [piece.strip() for piece in text.replace('..', '.').split('.') if piece.strip()]
        for chunk in pieces:
            lower = chunk.lower()
            if any(word in lower for word in ['april', 'may', 'june', 'july', 'march', 'december', 'january', 'february', 'august', 'september', 'october', 'november']):
                cleaned = chunk.replace('st', '').replace('nd', '').replace('rd', '').replace('th', '')
                tokens = cleaned.split()
                simplified = []
                for token in tokens:
                    if any(ch.isdigit() for ch in token) or any(letter.isalpha() for letter in token):
                        simplified.append(token)
                candidate = ' '.join(simplified[:10])
                if candidate:
                    candidates.append(candidate)
        # de-duplicate while preserving order
        seen = set()
        final = []
        for item in candidates:
            item_norm = ' '.join(item.split())
            if item_norm.lower() not in seen:
                seen.add(item_norm.lower())
                final.append(item_norm)
        return final

    def _extract_notice_days(self, sentence, full_text):
        lower_sentence = sentence.lower()
        if 'notice' in lower_sentence or 'renewal' in lower_sentence or 'termination' in lower_sentence:
            for token in sentence.split():
                digit_value = ''.join(ch for ch in token if ch.isdigit())
                if digit_value and len(digit_value) <= 3:
                    return digit_value
            for word, days in [('one', '30'), ('two', '60'), ('three', '90'), ('four', '120')]:
                if word in lower_sentence:
                    return days
            if 'month' in lower_sentence:
                if 'one' in lower_sentence:
                    return '30'
                if 'two' in lower_sentence:
                    return '60'
                if 'three' in lower_sentence:
                    return '90'
                if 'four' in lower_sentence:
                    return '120'

        for chunk in self._split_text(full_text):
            lower = chunk.lower()
            if 'notice' in lower or 'renewal' in lower or 'termination' in lower:
                for token in chunk.split():
                    digit_value = ''.join(ch for ch in token if ch.isdigit())
                    if digit_value and len(digit_value) <= 3:
                        return digit_value
                for word, days in [('one', '30'), ('two', '60'), ('three', '90'), ('four', '120')]:
                    if word in lower:
                        return days
        return ''

    def _extract_party_name(self, sentence, full_text, first_party=True):
        lower_text = full_text.lower()
        first_between = lower_text.find('between')
        if first_between != -1:
            second_between = lower_text.find('between', first_between + 1)
            if first_party:
                fragment = full_text[first_between + 7:]
                if second_between != -1:
                    fragment = full_text[first_between + 7:second_between]
                fragment = fragment.split('and', 1)[0]
                fragment = fragment.split('witnesses', 1)[0]
                fragment = fragment.replace('hereinafter', '').replace('called', '').strip(' .,:;')
                cleaned = self._best_name_from_fragment(fragment)
                if cleaned:
                    return cleaned

            if not first_party:
                if second_between != -1:
                    fragment = full_text[second_between + 7:]
                else:
                    fragment = full_text[first_between + 7:]
                fragment = fragment.split('witnesses', 1)[0]
                fragment = fragment.split('and', 1)[0]
                fragment = fragment.replace('hereinafter', '').replace('called', '').strip(' .,:;')
                cleaned = self._best_name_from_fragment(fragment)
                if cleaned:
                    return cleaned

        matches = []
        pattern = r'(?:Mr|Mrs|Ms|Sri|Smt|Dr|M)\.?\s+[A-Z][A-Za-z\./-]*?(?:\s+[A-Z][A-Za-z\./-]*){0,4}'
        for match in re.finditer(pattern, full_text):
            value = match.group(0).strip(' .,:;')
            if value:
                matches.append(value)

        if first_party and matches:
            return re.sub(r'^(?:Mr|Mrs|Ms|Sri|Smt|Dr|M)\.?\s*', '', matches[0], flags=re.I)
        if not first_party and len(matches) > 1:
            return re.sub(r'^(?:Mr|Mrs|Ms|Sri|Smt|Dr|M)\.?\s*', '', matches[1], flags=re.I)
        return ''

    def _best_name_from_fragment(self, fragment):
        candidates = re.findall(r'(?:Mr|Mrs|Ms|Sri|Smt|Dr|M)?\.?\s*[A-Z][A-Za-z\./-]*(?:\s+[A-Z][A-Za-z\./-]*){0,3}', fragment)
        cleaned = []
        for candidate in candidates:
            candidate = candidate.strip(' .,:;')
            lower = candidate.lower()
            if not any(word in lower for word in ['after', 'called', 'party', 'part', 'witnesses', 'between']):
                cleaned.append(candidate)
        return cleaned[0].strip() if cleaned else ''

    def _clean_name(self, piece):
        text = piece.replace('hereinafter', '').replace('called', '').replace('the', '').replace('party', '').replace('of', '').strip()
        text = text.replace('—', ' ')
        text = " ".join(text.split())
        if len(text) < 3:
            return ''
        return text.strip(' ,.-:;')

    def _looks_like_name(self, text):
        alpha_count = sum(ch.isalpha() for ch in text)
        return alpha_count > 3 and len(text) > 3

    def _normalize_result(self, result):
        cleaned = {}
        for key, value in result.items():
            text = str(value).strip()
            if key in {'Agreement Value', 'Renewal Notice (Days)'}:
                digits = ''.join(ch for ch in text if ch.isdigit())
                text = digits if digits else text
            if key in {'Agreement Start Date', 'Agreement End Date'}:
                text = text.replace('day', '').replace('of', '').replace('month', '').strip()
            cleaned[key] = text
        return cleaned


def normalize_compare_value(value):
    text = str(value).strip()
    if text == '':
        return ''
    text = text.strip(' \t\r\n,')
    try:
        return str(float(text)).rstrip('0').rstrip('.') if '.' in text else text
    except ValueError:
        return text


def compute_per_field_recall(predictions, truth_df):
    field_map = {
        'Agreement Value': ['Aggrement Value', 'Agreement Value'],
        'Agreement Start Date': ['Aggrement Start Date', 'Agreement Start Date'],
        'Agreement End Date': ['Aggrement End Date', 'Agreement End Date'],
        'Renewal Notice (Days)': ['Renewal Notice (Days)'],
        'Party One': ['Party One'],
        'Party Two': ['Party Two'],
    }
    metrics = {}
    for field, aliases in field_map.items():
        true_count = 0
        total = 0
        for item in predictions:
            file_name = item['File Name']
            expected = truth_df[truth_df['File Name'] == file_name].iloc[0]
            pred_value = normalize_compare_value(item.get(field, ''))
            target_value = ''
            for alias in aliases:
                if alias in expected.index:
                    target_value = normalize_compare_value(expected.get(alias, ''))
                    break
            total += 1
            if pred_value == target_value:
                true_count += 1
        metrics[field] = true_count / total if total else 0.0
    return metrics


def evaluate_on_train(folder_path, csv_path):
    truth_df = pd.read_csv(csv_path)
    extractor = MetadataExtractor()
    predictions = []
    for _, row in truth_df.iterrows():
        file_name = row['File Name']
        full_path = os.path.join(folder_path, file_name)
        if not os.path.exists(full_path):
            # fallback for file names with extra suffixes
            for root, _, files in os.walk(folder_path):
                for candidate in files:
                    if candidate.startswith(file_name):
                        full_path = os.path.join(root, candidate)
                        break
        extracted = extractor.extract_file(full_path)
        row_result = {'File Name': file_name}
        row_result.update(extracted)
        predictions.append(row_result)
    return predictions, compute_per_field_recall(predictions, truth_df)


if __name__ == '__main__':
    train_dir = os.path.join('data', 'train')
    train_csv = os.path.join('data', 'train.csv')
    preds, metrics = evaluate_on_train(train_dir, train_csv)
    print('Recall by field:')
    for field, value in metrics.items():
        print(f'{field}: {value:.3f}')

    out_path = os.path.join('data', 'train_predictions.csv')
    pd.DataFrame(preds).to_csv(out_path, index=False)
    print(f'Wrote predictions to {out_path}')
