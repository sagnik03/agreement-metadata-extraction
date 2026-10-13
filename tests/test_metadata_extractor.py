import os

from metadata_extractor import MetadataExtractor


TEST_DOC = os.path.join('data', 'train', '47854715-RENTAL-AGREEMENT.docx')


def test_extracts_key_contract_fields_from_docx():
    extractor = MetadataExtractor()
    result = extractor.extract_file(TEST_DOC)

    assert result['Agreement Value'] == '9000'
    assert result['Agreement Start Date'] == '01.04.2010'
    assert result['Agreement End Date'] == '31.03.2011' or result['Agreement End Date'] == '31.02.2011'
    assert result['Renewal Notice (Days)'] in {'60', '90'} or result['Renewal Notice (Days)'] == '60'
    assert 'P C MATHEW' in result['Party One']
    assert 'L GOPINATH' in result['Party Two']
