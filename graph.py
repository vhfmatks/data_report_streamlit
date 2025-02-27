from typing import Annotated, Sequence, TypedDict, Union
from langchain_core.messages import BaseMessage
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
import operator
from pydantic import BaseModel
from typing import List, Dict, Any
from chains import (
    get_draft_chain, get_analysis_plan_chain, analyze_step,
    get_interim_report_chain, get_chart_plan_chain,
    get_chart_code_generate_chain, get_chart_code_check_chain,
    generate_chart, init_database, get_sql_chain,
    get_execute_query_chain, get_answer_chain
)

# 상태 정의
class AnalysisState(TypedDict):
    title: str
    purpose: str
    schema: str
    draft: str
    analysis_plan: Dict
    analysis_results: List[Dict]
    interim_report: str
    chart_plan: Dict
    chart_results: List[Dict]
    final_report: str
    error: Union[str, None]
    current_step: str

# 노드 정의
def draft_analysis(state: AnalysisState, llm) -> AnalysisState:
    """분석 계획서 초안 작성 노드"""
    try:
        draft_chain = get_draft_chain(llm)
        state["draft"] = draft_chain.invoke({
            "title": state["title"],
            "purpose": state["purpose"]
        })
        state["current_step"] = "analysis_plan"
    except Exception as e:
        state["error"] = str(e)
        state["current_step"] = "error"
    return state

def create_analysis_plan(state: AnalysisState, llm) -> AnalysisState:
    """상세 분석 계획 작성 노드"""
    try:
        analysis_plan_chain = get_analysis_plan_chain(llm)
        state["analysis_plan"] = analysis_plan_chain.invoke({
            "title": state["title"],
            "purpose": state["purpose"],
            "draft": state["draft"],
            "schema": state["schema"]
        })
        state["current_step"] = "execute_analysis"
    except Exception as e:
        state["error"] = str(e)
        state["current_step"] = "error"
    return state

def execute_analysis(state: AnalysisState, llm) -> AnalysisState:
    """분석 실행 노드"""
    try:
        db = init_database()
        sql_chain = get_sql_chain(llm, db)
        execute_chain = get_execute_query_chain(db)
        answer_chain = get_answer_chain(llm)
        
        analysis_results = []
        for step in state["analysis_plan"]["analysis_plan"]:
            result = analyze_step(
                step["description"],
                sql_chain,
                execute_chain,
                answer_chain
            )
            analysis_results.append({
                "step": step["step"],
                **result
            })
        
        state["analysis_results"] = analysis_results
        state["current_step"] = "interim_report"
    except Exception as e:
        state["error"] = str(e)
        state["current_step"] = "error"
    return state

def create_interim_report(state: AnalysisState, llm) -> AnalysisState:
    """중간보고서 작성 노드"""
    try:
        interim_report_chain = get_interim_report_chain(llm)
        state["interim_report"] = interim_report_chain.invoke({
            "title": state["title"],
            "purpose": state["purpose"],
            "analysis_results": state["analysis_results"],
            "schema": state["schema"]
        })
        state["current_step"] = "chart_plan"
    except Exception as e:
        state["error"] = str(e)
        state["current_step"] = "error"
    return state

def create_chart_plan(state: AnalysisState, llm) -> AnalysisState:
    """차트 계획 작성 노드"""
    try:
        chart_plan_chain = get_chart_plan_chain(llm)
        state["chart_plan"] = chart_plan_chain.invoke({
            "interim_report": state["interim_report"],
            "title": state["title"],
            "purpose": state["purpose"],
            "schema": state["schema"]
        })
        state["current_step"] = "generate_charts"
    except Exception as e:
        state["error"] = str(e)
        state["current_step"] = "error"
    return state

def generate_charts(state: AnalysisState, llm) -> AnalysisState:
    """차트 생성 노드"""
    try:
        from langchain_experimental.tools import PythonREPLTool
        
        db = init_database()
        sql_chain = get_sql_chain(llm, db)
        execute_chain = get_execute_query_chain(db)
        chart_code_generate_chain = get_chart_code_generate_chain(llm)
        chart_code_check_chain = get_chart_code_check_chain(llm)
        python_repl_tool = PythonREPLTool()
        
        chart_results = []
        for chart in state["chart_plan"]["charts"]:
            result = generate_chart(
                chart,
                sql_chain,
                execute_chain,
                chart_code_generate_chain,
                chart_code_check_chain,
                python_repl_tool
            )
            if result["success"]:
                chart_results.append({
                    "chart": chart,
                    **result
                })
        
        state["chart_results"] = chart_results
        state["current_step"] = "final_report"
    except Exception as e:
        state["error"] = str(e)
        state["current_step"] = "error"
    return state

def create_final_report(state: AnalysisState, llm) -> AnalysisState:
    """최종 보고서 작성 노드"""
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        final_report_prompt = PromptTemplate(
            template="""당신은 데이터 분석 전문가입니다.
            지금까지의 분석 결과와 차트를 바탕으로 최종 보고서를 작성해주세요.
            
            분석 주제: {title}
            분석 목적: {purpose}
            분석 데이터 스키마:
            {schema}
            
            중간 보고서:
            {interim_report}
            
            차트 분석 결과:
            {chart_results}
            
            다음 형식으로 최종 보고서를 작성해주세요:
            
            # 최종 보고서
            
            ## 1. 분석 개요
            - 분석 배경 및 목적
            - 분석 방법론
            
            ## 2. 주요 발견사항
            - 핵심 인사이트
            - 데이터 패턴
            - 시각화 결과
            
            ## 3. 결론 및 제언
            - 분석 결과 요약
            - 시사점
            - 향후 과제
            """,
            input_variables=["title", "purpose", "schema", "interim_report", "chart_results"]
        )
        
        final_report_chain = final_report_prompt | llm | StrOutputParser()
        
        chart_results_text = []
        for result in state["chart_results"]:
            chart = result["chart"]
            chart_results_text.append(f"""
            차트명: {chart["chart_name"]}
            차트 설명: {chart["chart_description"]}
            차트 유형: {chart["chart_type"]}
            강조 포인트:
            {chr(10).join([f'- {point}' for point in chart["emphasis_points"]])}
            """)
        
        state["final_report"] = final_report_chain.invoke({
            "title": state["title"],
            "purpose": state["purpose"],
            "schema": state["schema"],
            "interim_report": state["interim_report"],
            "chart_results": "\n".join(chart_results_text)
        })
        state["current_step"] = "end"
    except Exception as e:
        state["error"] = str(e)
        state["current_step"] = "error"
    return state

# 엣지 정의
def define_edges(state: AnalysisState) -> str:
    """현재 상태에 따라 다음 단계 결정"""
    if state["error"]:
        return "error"
    return state["current_step"]

# 그래프 생성
def create_analysis_graph(llm):
    """분석 워크플로우 그래프 생성"""
    # 워크플로우 그래프 정의
    workflow = StateGraph(AnalysisState)
    
    # 노드 추가
    workflow.add_node("draft", lambda x: draft_analysis(x, llm))
    workflow.add_node("analysis_plan", lambda x: create_analysis_plan(x, llm))
    workflow.add_node("execute_analysis", lambda x: execute_analysis(x, llm))
    workflow.add_node("interim_report", lambda x: create_interim_report(x, llm))
    workflow.add_node("chart_plan", lambda x: create_chart_plan(x, llm))
    workflow.add_node("generate_charts", lambda x: generate_charts(x, llm))
    workflow.add_node("final_report", lambda x: create_final_report(x, llm))
    
    # 엣지 설정
    workflow.add_edge("draft", "analysis_plan")
    workflow.add_edge("analysis_plan", "execute_analysis")
    workflow.add_edge("execute_analysis", "interim_report")
    workflow.add_edge("interim_report", "chart_plan")
    workflow.add_edge("chart_plan", "generate_charts")
    workflow.add_edge("generate_charts", "final_report")
    
    # 종료 조건 설정
    workflow.set_entry_point("draft")
    workflow.add_terminal_node("end")
    workflow.add_terminal_node("error")
    
    return workflow.compile() 