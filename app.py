import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from chains import (
    init_database, get_sql_chain, get_execute_query_chain, get_answer_chain, analyze_step, get_interim_report_chain,
    get_chart_plan_chain, get_chart_code_generate_chain, get_chart_code_check_chain, generate_chart,
    get_final_report_chain, get_query_check_chain
)
from langchain_core.prompts import PromptTemplate
import pandas as pd
from langchain_experimental.tools import PythonREPLTool
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="데이터 분석 PoC",
    page_icon="📊",
    layout="wide"
)

# 제목
st.title("📊 데이터 분석 PoC")

# 스키마 입력
with open('schema.md', 'r', encoding='utf-8') as f:
    schema_content = f.read()
schema = st.text_area("스키마", value=schema_content, height=100)

# Pydantic 모델 정의
class DataAnalysisStep(BaseModel):
    step: str = Field(description="데이터 분석 단계- 숫자와 해당 단계에 대한 title(예: 1. 성별 매출 분석)")
    description: str = Field(description="데이터 분석 단계 설명(prompt)")

class DataAnalysisPlan(BaseModel):
    analysis_plan: List[DataAnalysisStep] = Field(description="데이터 분석 단계 리스트")

# LLM 초기화
@st.cache_resource
def get_llm(model_name="qwen-2.5-32b", temperature=0.7):
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
    )

# 초안 작성 체인 초기화
@st.cache_resource
def get_draft_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 데이터 분석 전문가입니다. 
        주어진 분석 주제와 목적을 바탕으로 데이터 분석 계획서 초안을 작성해주세요.
        
        다음 형식으로 작성해주세요:
        
        # 1. 분석 개요
        - 분석 주제와 목적 요약
        - 기대 효과
        
        # 2. 분석 방법론
        - 필요한 데이터
        - 주요 분석 기법
        - 예상되는 인사이트
        
        # 3. 실행 계획
        - 단계별 수행 계획
        - 예상 소요 시간
        
        # 4. 고려사항
        - 제약사항
        - 리스크 요인
        """),
        ("human", "분석 주제: {title}\n분석 목적: {purpose}")
    ])
    
    return prompt | get_llm() | StrOutputParser()

# 상세 분석 계획 체인 초기화
@st.cache_resource
def get_analysis_plan_chain():
    parser = JsonOutputParser(pydantic_object=DataAnalysisPlan)
    format_instructions = parser.get_format_instructions()
    prompt = PromptTemplate(
        template="""
        당신은 10년차 데이터분석 전문가 입니다. 
        주어진 데이터분석보고서 초안과 스키마를 가지고 데이터 분석 계획을 세워주세요.
        이번 분석 계획은 "데이터 탐색 및 분석" 에 초점을 맞추어 진행할 예정입니다.
        데이터준비,파악,시각화 등은 이번 계획에 포함하지 않습니다.
        데이터 분석 시, 독립변수의 복합적인 관계에서 인사이트를 도출하는 것을 목표로 합니다.
        최소 5개 이상의 단계를 세워주세요.
        
        ** 주의 **
         - 각 단계는 SQL 을 작성하여 진행할 예정입니다.
         - SQL 작성할 수 있는 분석계획을 만들어주세요
         - step 작성 시 예시를 참고하여 작성해주세요. (번호 및 단계에 대한 title포함)
            예시 : 
            1. 성별 매출 분석
            2. 시간대별, 지역별 매출 추이 분석
            3. 업종별 선호도 분석을 통한 마케팅 전략 수립
            4. 고객 세그먼트별 맞춤형 서비스 제안

        분석보고서초안 : {draft}
        분석보고서제목 : {title}
        분석목적 : {purpose}
        Schema: {schema}
        Answer:

        Please response in Json format.
        {format_instructions}
        """
        , input_variables=["draft", "title", "purpose", "schema"]
        , partial_variables={"format_instructions": format_instructions}
    )
    
    return prompt | get_llm() | parser

# 데이터베이스 및 체인 초기화
@st.cache_resource
def init_chains():
    db = init_database()
    llm = get_llm()
    sql_chain = get_sql_chain(llm, db)
    query_check_chain = get_query_check_chain(get_llm(temperature=0.0))
    execute_chain = get_execute_query_chain(db)
    answer_chain = get_answer_chain(llm)
    return sql_chain, query_check_chain, execute_chain, answer_chain

# 데이터 분석 목적과 주제 입력 폼
with st.form(key="analysis_form"):
    st.subheader("📝 데이터 분석 정보")
    
    analysis_title = st.text_input(
        "분석 주제",
        value="축제 전후 상권 분석",
        placeholder="예: 2024년 1분기 매출 분석"
    )
    analysis_purpose = st.text_area(
        "분석 목적",
        value="""
2023년 2월~3월 동안 실시된 지역 축제를 통해
지역 상권의 매출에 변화가 있었는지 알아보기 위함
""",
        placeholder="예: 성별, 연령대별 매출 현황을 파악하여 마케팅 전략 수립",
        height=100
    )
    
    submit_button = st.form_submit_button("분석 시작")

if submit_button:
    # 입력된 정보 표시
    st.sidebar.subheader("📋 분석 개요")
    st.sidebar.markdown(f"""
    **분석 주제:** {analysis_title}
    
    **분석 목적:** {analysis_purpose}
    """)
    
    # 초안 작성
    with st.spinner("분석 계획서 초안을 작성 중입니다..."):
        draft_chain = get_draft_chain()
        analysis_draft = draft_chain.invoke({
            "title": analysis_title,
            "purpose": analysis_purpose
        })
        
        st.markdown("## 📑 분석 계획서 초안")
        st.markdown(analysis_draft)
        
        # 상세 분석 계획 작성
        with st.spinner("상세 데이터 분석 계획을 작성 중입니다..."):
            analysis_plan_chain = get_analysis_plan_chain()
            analysis_plan = analysis_plan_chain.invoke({
                "title": analysis_title,
                "purpose": analysis_purpose,
                "draft": analysis_draft,
                "schema": schema
            })
            
            st.markdown("## 📊 상세 데이터 분석 계획")
            
            # 분석 계획을 단계별로 표시
            sql_chain, query_check_chain, execute_chain, answer_chain = init_chains()
            analysis_results = []

            for step in analysis_plan["analysis_plan"]:
                with st.expander(f"### {step['step']}"):
                    st.markdown(step['description'])
                    
                    with st.spinner(f"{step['step']} 분석 중..."):
                        result = analyze_step(
                            step['description'],
                            sql_chain,
                            query_check_chain,
                            execute_chain,
                            answer_chain
                        )
                        
                        if "error" in result:
                            st.error(f"분석 중 오류 발생: {result['error']}")
                        else:
                            st.subheader("🔍 SQL 쿼리")
                            st.code(result["sql_query"], language="sql")
                            
                            st.subheader("📊 분석 결과")
                            # st.code(str(result["result"]), language="json")
                            try:
                                df = pd.DataFrame.from_dict(eval(result["result"]))
                                st.dataframe(df)
                            except:
                                st.error("데이터프레임 변환 중 오류가 발생했습니다.")
                            st.subheader("💡 해석")
                            st.markdown(result["answer"])
                        
                        analysis_results.append({
                            "step": step['step'],
                            **result
                        })
            if analysis_results:
                # 분석 결과 다운로드 버튼
                with st.spinner("중간보고서 작성 중..."):
                    interim_report_chain = get_interim_report_chain(get_llm())
                    interim_report = interim_report_chain.invoke({
                        "title": analysis_title,
                        "purpose": analysis_purpose,
                        "analysis_results": analysis_results,
                        "schema": schema
                    })
                    st.markdown(interim_report)

                ## 중간 보고서를 이용해 차트 생성
                st.subheader("💡 차트 생성")
                chart_gen_steps = None

                with st.spinner("차트 생성 계획 중..."):
                    chart_plan_chain = get_chart_plan_chain(get_llm())
                    chart_gen_steps = chart_plan_chain.invoke({
                        "interim_report": interim_report,
                        "title": analysis_title,
                        "purpose": analysis_purpose,
                        "schema": schema
                    })
                
                with st.spinner("차트 생성 중..."):
                    python_repl_tool = PythonREPLTool()
                    chart_code_generate_chain = get_chart_code_generate_chain(get_llm())
                    chart_code_check_chain = get_chart_code_check_chain(get_llm())

                    
                    for chart in chart_gen_steps['charts']:
                        st.markdown(f"### 📈 {chart['chart_name']}")
                        st.markdown(f"**설명:** {chart['chart_description']}")
                        st.markdown(f"**차트 유형:** {chart['chart_type']}")
                        st.markdown("**강조 포인트:**")
                        for point in chart['emphasis_points']:
                            st.markdown(f"- {point}")
                            
                        with st.spinner(f"{chart['chart_name']} 생성 중..."):
                            result = generate_chart(
                                chart,
                                sql_chain,
                                execute_chain,
                                chart_code_generate_chain,
                                chart_code_check_chain,
                                python_repl_tool
                            )
                            
                            if result["success"]:
                                st.markdown("**SQL 쿼리:**")
                                st.code(result["sql_query"], language="sql")
                                
                                st.markdown("**차트 코드:**")
                                st.code(result["chart_code"], language="python")
                                
                                # exec(result["chart_code"], globals())
                            else:
                                st.error(f"차트 생성 중 오류 발생: {result['error']}")
                                continue

                    st.markdown("## 📊 최종 보고서")
                    
                    with st.spinner("최종 보고서 작성 중..."):
                        final_report_chain = get_final_report_chain(get_llm())
                        
                        chart_results = []
                        for chart in chart_gen_steps['charts']:
                            chart_results.append(f"""
                            차트명: {chart['chart_name']}
                            차트 설명: {chart['chart_description']}
                            차트 유형: {chart['chart_type']}
                            강조 포인트:
                            {chr(10).join([f'- {point}' for point in chart['emphasis_points']])}
                            """)
                        
                        final_report = final_report_chain.invoke({
                            "title": analysis_title,
                            "purpose": analysis_purpose,
                            "schema": schema,
                            "interim_report": interim_report,
                            "chart_results": "\n".join(chart_results)
                        })
                        
                        st.markdown(final_report)
