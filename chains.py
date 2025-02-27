from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from typing import List, Dict
from pydantic import BaseModel, Field

def init_database(db_path: str = "sqlite:///data.db") -> SQLDatabase:
    """데이터베이스 초기화 함수"""
    return SQLDatabase.from_uri(db_path)

def get_sql_chain(llm, db: SQLDatabase):
    """SQL 쿼리 생성 체인"""
    prompt = PromptTemplate(
        template="""
        Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results.
        Use the following format:
        Please response Just SQL Query, not any other text. 
        do not include any other text except SQL Query. like "Here is the SQL query:" or "SQL Query:" or "sql"
        
        Only use the following tables:
        {table_info}

        Question: {input}""",
        input_variables=["input", "table_info", "dialect"],
        partial_variables={"top_k": 10}
    )
    
    return create_sql_query_chain(llm, db, prompt)

def get_query_check_chain(llm) :

    class QueryCheck(BaseModel):
        query: str = Field(description="SQL 쿼리")
        description: str = Field(description="SQL 쿼리 설명")

    parser = JsonOutputParser(pydantic_object=QueryCheck)
    format_instructions = parser.get_format_instructions()
    
    query_check_prompt =  PromptTemplate(
        template="""You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.

User Question: {description}
SQLQuery: {query}

Please response in Json format.
{format_instructions}
""",
        input_variables=["query", "description"],
        partial_variables={"format_instructions": format_instructions}
    )
    return query_check_prompt | llm | parser

def get_execute_query_chain(db: SQLDatabase):
    """SQL 실행 체인"""
    return QuerySQLDatabaseTool(
        db=db,
        return_format="json"
    )

def get_answer_chain(llm):
    """결과 해석 체인"""
    prompt = PromptTemplate.from_template(
        """
        Given the results of the SQL query, return the answer to the question.

        Question: {question}
        SQLQuery: {query}
        SQLResult: {result}
        Answer:
        """
    )
    
    return prompt | llm | StrOutputParser()

def analyze_step(step_description: str, sql_chain, query_check_chain, execute_chain, answer_chain):
    """각 분석 단계 실행 함수"""
    try:
        # SQL 쿼리 생성
        generated_sql_query = sql_chain.invoke({
            "question": step_description
        })
        # print(generated_sql_query)

        # generated_sql_query = query_check_chain.invoke(
        #     {
        #         "query": generated_sql_query,
        #         "description": step_description
        #     }
        # )
        # print(generated_sql_query)
        sql = generated_sql_query

        # 쿼리 실행
        sql_result = execute_chain.invoke({
            "query": sql
        })
        
        # 결과 해석
        answer = answer_chain.invoke({
            "question": step_description,
            "query": sql,
            "result": sql_result
        })
        
        return {
            "sql_query": sql,
            "result": sql_result,
            "answer": answer
        }
    except Exception as e:
        return {
            "error": str(e)
        }

def get_interim_report_chain(llm):
    """중간보고서 작성 체인"""
    prompt = PromptTemplate(
        template="""당신은 데이터 분석 전문가입니다. 
        지금까지의 분석 결과를 바탕으로 중간보고서를 작성해주세요.
        보고서는 한국어로 작성해주세요.
        
        분석 주제: {title}
        분석 목적: {purpose}
        분석 데이터 스키마 :
        {schema}
        
        분석 단계별 결과:
        {analysis_results}
        
        다음 형식으로 중간보고서를 작성해주세요:
        
        # 중간보고서
        
        ## 1. 분석 개요
        - 분석 배경 및 목적
        - 현재까지의 진행 상황
        
        ## 2. 주요 발견사항
        - 핵심 인사이트
        - 데이터 패턴
        - 특이사항
        
        ## 3. 중간 결론
        - 현재까지의 분석 결과 요약
        - 시사점
        
        ## 4. 향후 계획
        - 추가 분석이 필요한 영역
        - 개선사항 및 제안
        """
        , input_variables=["title", "purpose", "schema", "analysis_results"]
    )
    
    return prompt | llm | StrOutputParser()

class ReportChartItem(BaseModel):
    chart_name: str = Field(description="차트 이름")
    chart_description: str = Field(description="차트 설명")
    chart_type: str = Field(description="차트 유형")
    sql_query: str = Field(description="SQL 쿼리")
    chart_generate_prompt: str = Field(description="차트 생성 프롬프트")
    emphasis_points: List[str] = Field(description="강조할 포인트")
    chart_png_file_name: str = Field(description="차트 png 파일 이름")

class ReportChart(BaseModel):
    charts: List[ReportChartItem] = Field(description="차트 리스트")

def get_chart_plan_chain(llm):
    parser = JsonOutputParser(pydantic_object=ReportChart)
    format_instructions = parser.get_format_instructions()
    
    chart_plan_prompt = PromptTemplate(
        template="""
        당신은 데이터 시각화 전문가입니다.
        중간보고서를 바탕으로 효과적인 차트 계획을 수립해주세요.
        중간보고서에서 주장하고자 하는 바를 뒷받침하는 차트를 수립해주세요.
        
        ** 요구사항 **
        1. 중간보고서의 주요 발견사항과 결론을 시각화로 표현해주세요.
        2. 각 차트는 고유한 인사이트를 제공해야 합니다.
        3. 데이터 비교가 필요한 경우 적절한 비교 차트를 사용하세요.
        4. 금액 데이터는 천원 단위로 표시합니다.
        5. 지역 데이터는 계층적으로 분석합니다.
        
        ** 차트 유형 **
        1. 막대 차트: 범주형 데이터 비교
        2. 선 차트: 시계열 데이터 및 추세
        3. 원 차트: 비율 및 구성
        4. 산점도: 상관관계 분석
        5. 히트맵: 복잡한 관계성 표현
        6. 박스플롯: 분포 및 이상치
        
        중간보고서: {interim_report}
        분석 주제: {title}
        분석 목적: {purpose}
        스키마: {schema}
        
        Please response in Json format.
        {format_instructions}
        """,
        input_variables=["interim_report", "title", "purpose", "schema"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    return chart_plan_prompt | llm | parser

def get_chart_code_generate_chain(llm):
    chart_code_generate_prompt = PromptTemplate(
        template="""
        당신은 데이터 시각화 전문가입니다.
        주어진 데이터를 시각화하는 Python 코드를 작성해주세요.
        
        차트 요구사항: {chart_generate_prompt}
        차트 이름: {chart_name}
        SQL 쿼리: {sql}
        데이터: {data}
        
        ** 주의 **
        - 주석, 설명은 생략 : 오직 Python 코드만 작성
        - 다음 라이브러리를 사용하여 코드를 작성해주세요:
        ```python
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        import json
        from plotly.subplots import make_subplots
        ```
        
        요구사항:
        1. 데이터프레임 생성 시 json.loads()를 사용하여 문자열을 딕셔너리로 변환
        2. 차트는 plotly를 사용하여 생성
        3. 차트 제목과 축 레이블을 명확하게 지정
        4. 적절한 색상과 테마 사용
        5. 필요한 경우 데이터 전처리 수행
        6. 차트생성 후에는 streamlit 에 보여줄 수 있도록 해주세요. (예시 : st.plotly_chart(fig))
        
        Python 코드만 출력해주세요.
        """,
        input_variables=["chart_generate_prompt", "chart_name", "sql", "data", "chart_png_file_name"]
    )
    
    return chart_code_generate_prompt | llm | StrOutputParser()

def get_chart_code_check_chain(llm):
    class ChartCodeCheck(BaseModel):
        code: str = Field(description="Python 코드(주석, 설명, 설명은 생략 : 오직 Python 코드만 작성)")
        description: str = Field(description="코드 또는 수정사항 설명")
    parser = JsonOutputParser(pydantic_object=ChartCodeCheck)
    format_instructions = parser.get_format_instructions()
    chart_code_check_prompt = PromptTemplate(
        template="""
        다음 Python 코드를 검토하고 개선해주세요.
        code 가 오류없이 실행되는지 확인하고, 오류발생 가능성이 있으면 수정하기 바랍니다.
        
        차트 요구사항: {chart_generate_prompt}
        
        코드:
        {code}
        
        다음 사항을 확인하고 개선해주세요:
        1. 코드 실행 가능 여부
        2. 차트 스타일링 적절성
        3. 데이터 처리 정확성
        4. 시각화 효과성
        5. streamlit 호환성
        6. 항상 마지막 라인에는 streamlit에 보여줄 수 있는지 확인해주세요 (예시 : st.plotly_chart(fig))
        
        개선된 Python 코드만 출력해주세요.

        Please response in Json format.
        {format_instructions}
        """,
        input_variables=["chart_generate_prompt", "code"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    return chart_code_check_prompt | llm | parser

def generate_chart(chart, sql_chain, execute_chain, chart_code_generate_chain, chart_code_check_chain, python_repl_tool):
    try:
        # SQL 쿼리 생성 및 실행
        generated_sql_query = sql_chain.invoke({
            "question": f"{chart['chart_description']} 를 위한 SQL 쿼리를 작성해주세요. 참고: {chart['sql_query']}"
        })
        
        sql_result = execute_chain.invoke({
            "query": generated_sql_query
        })
        
        # 차트 코드 생성
        chart_code = chart_code_generate_chain.invoke({
            "chart_generate_prompt": f"차트 이름 : {chart['chart_name']} 차트 설명 : {chart['chart_description']} 차트 유형 : {chart['chart_type']} 차트 강조 포인트 : {chart['emphasis_points']} 차트 생성 프롬프트 : {chart['chart_generate_prompt']}",
            "chart_name": chart['chart_name'],
            "sql": generated_sql_query,
            "data": sql_result,
            "chart_png_file_name": chart['chart_png_file_name']
        })
        
        # 코드 검증 및 개선
        final_code = chart_code_check_chain.invoke({
            "chart_generate_prompt": f"차트 이름 : {chart['chart_name']} 차트 설명 : {chart['chart_description']} 차트 유형 : {chart['chart_type']} 차트 강조 포인트 : {chart['emphasis_points']} 차트 생성 프롬프트 : {chart['chart_generate_prompt']}",
            "code": chart_code
        })
        print(final_code)
        
        return {
            "sql_query": generated_sql_query,
            "chart_code": final_code['code'],
            "success": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

def get_final_report_chain(llm):
    """최종 보고서 작성 체인"""
    prompt = PromptTemplate(
        template="""
        당신은 전문 데이터 분석가입니다. 주어진 데이터를 바탕으로 포괄적인 데이터 분석 보고서를 작성하세요. 보고서는 다음과 같은 내용을 포함해야 합니다.

### 분석 개요
- **분석 주제:** {title}
- **분석 목적:** {purpose}

### 데이터 개요
- **데이터 설명:** 주어진 데이터 스키마({schema})를 바탕으로, 데이터의 특성과 변수를 분석하세요.
- **데이터 품질 검토:** 데이터 누락, 이상값, 데이터 분포 등을 분석하여 데이터의 신뢰성을 평가하세요.

### 탐색적 데이터 분석 (EDA)
- **기술 통계:** 주요 변수의 평균, 중앙값, 최댓값, 최솟값, 표준 편차 등을 포함한 기술 통계 분석을 수행하세요.
- **변수 간 관계 분석:** 주요 변수 간의 상관관계 및 패턴을 설명하세요.
- **시각적 분석:** 차트 및 그래프({chart_results})를 분석하여 데이터의 주요 트렌드를 설명하세요.

### 데이터 모델링 및 인사이트
- **주요 발견사항:** 데이터를 기반으로 도출된 주요 인사이트를 서술하세요.
- **패턴 및 추세 분석:** 데이터에서 발견된 의미 있는 패턴과 추세를 설명하세요.
- **예측 분석 (필요한 경우):** 주어진 데이터가 적절하다면, 간단한 예측 모델을 사용하여 향후 트렌드를 예측하세요.

### 결론 및 추천 사항
- **분석 결과 요약:** 주요 분석 결과를 요약하세요.
- **비즈니스 및 전략적 시사점:** 분석 결과를 기반으로 실용적인 인사이트 및 추천 사항을 제공하세요.
- **추가 분석 제안:** 추가적으로 수행할 수 있는 분석 방향을 제안하세요.

보고서는 전문적인 스타일을 유지하며, 명확하고 논리적으로 작성하세요.

        """,
        input_variables=["title", "purpose", "schema", "interim_report", "chart_results"]
    )
    
    return prompt | llm | StrOutputParser()
