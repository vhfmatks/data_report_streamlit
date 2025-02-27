# 데이터 스키마 정의

| 순번 | 컬럼명      | 표시명           | 설명                                                                                  | 데이터 타입 |
| ---- | ----------- | ---------------- | ------------------------------------------------------------------------------------- | ----------- |
| 1    | STRD_YYMM   | 기준년월         | 데이터 수집 기준 년월 (YYYYMM)                                                        | datetime    |
| 2    | STRD_DATE   | 기준일자         | 데이터 수집 기준 일자 (YYYYMMDD)                                                      | datetime    |
| 3    | WK_CD       | 요일구분         | 1: 평일, 2: 주말                                                                      | categorical |
| 4    | TIMC_CD     | 시간대구분       | 1: 새벽(00-06), 2: 오전(06-12), 3: 오후(12-18), 4: 야간(18-24)                        | categorical |
| 5    | MER_SIDO_CD | 가맹점시도코드   | 가맹점 소재지 시도 코드                                                               | categorical |
| 6    | MER_CCG_CD  | 가맹점시군구코드 | 가맹점 소재지 시군구 코드                                                             | categorical |
| 7    | MER_ADNG_CD | 가맹점행정동코드 | 가맹점 소재지 행정동 코드                                                             | categorical |
| 8    | BUZ_LCLS_NM | 업종대분류명     | 가맹점 업종 대분류                                                                    | categorical |
| 9    | BUZ_MCLS_NM | 업종중분류명     | 가맹점 업종 중분류                                                                    | categorical |
| 10   | BUZ_SCLS_NM | 업종소분류명     | 가맹점 업종 소분류                                                                    | categorical |
| 11   | CST_SIDO_CD | 고객시도코드     | 고객 거주지 시도 코드                                                                 | categorical |
| 12   | CST_CCG_CD  | 고객시군구코드   | 고객 거주지 시군구 코드                                                               | categorical |
| 13   | CST_ADNG_CD | 고객행정동코드   | 고객 거주지 행정동 코드                                                               | categorical |
| 14   | SEX_CD      | 성별코드         | 1: 남성, 2: 여성                                                                      | categorical |
| 15   | AGE_CD      | 연령대코드       | 20: 20대, 30: 30대, 40: 40대, 50: 50대, 60: 60대 이상                                 | categorical |
| 16   | HOSH_TYP_CD | 가구유형코드     | 1: 1인가구, 2: 부부가구, 3: 부부+자녀, 4: 한부모+자녀, 5: 3세대이상, 9: 기타          | categorical |
| 17   | INCM_NR_CD  | 소득구간코드     | 1: 3천만원 미만, 2: 3천-5천만원, 3: 5천-7천만원, 4: 7천-1억원, 5: 1억원 이상, 9: 기타 | categorical |
| 18   | AMT         | 매출금액         | 매출 발생 금액 (원)                                                                   | numeric     |
| 19   | CNT         | 매출건수         | 매출 발생 건수                                                                        | numeric     |
