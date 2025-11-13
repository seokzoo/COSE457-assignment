import sys
import os
from functools import lru_cache
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# .env 파일에서 API 키 로드
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    sys.exit(1)


@lru_cache(maxsize=1)
def get_llm():
    """LLM 초기화 - 앱 실행 시 한 번만 생성"""
    print("🤖 LLM을 초기화합니다... (OpenAI: gpt-4o-mini)")
    return ChatOpenAI(
        model="gpt-4o-mini",
        max_tokens=2000,
        temperature=0.7,
        streaming=True,
    )


@lru_cache(maxsize=1)
def get_retriever_from_pdf(pdf_path: str):
    """PDF 파일로부터 Knowledge Base Retriever 초기화"""
    if not os.path.exists(pdf_path):
        print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return None
    
    try:
        print(f"📚 PDF 문서를 로드합니다: {pdf_path}")
        # 1. 문서 로드
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # 2. 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 3. 임베딩 및 벡터 스토어 생성
        print("🧠 문서를 임베딩하고 벡터 스토어를 생성합니다... (FAISS)")
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(splits, embeddings)

        # 4. Retriever 반환
        return vector_store.as_retriever(
            search_kwargs={"k": 5} # 원본 코드의 numberOfResults=5와 동일
        )
    except Exception as e:
        print(f"❌ Retriever 초기화 실패: {e}")
        return None


def format_docs(docs):
    """검색된 문서를 문자열로 변환 (LangChain Document 객체용)"""
    if not docs:
        print("⚠️ 검색된 문서 없음")
        return "관련 문서를 찾을 수 없습니다."

    # LangChain의 Document 객체에서 page_content를 바로 추출
    formatted = [
        doc.page_content for doc in docs if hasattr(doc, "page_content") and doc.page_content
    ]

    result = (
        "\n\n---\n\n".join(formatted)
        if formatted
        else "문서 내용을 추출할 수 없습니다."
    )
    print(f"✅ {len(formatted)}개 문서 포맷 완료 (총 {len(result)}자)")
    return result


def create_chain_with_kb(retriever, llm):
    """RAG 체인 생성 - Retriever로 문서 검색 후 LLM에 전달"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """다음 문맥(context)을 참고하여 질문에 답변하세요.
문맥에 답이 없으면 모른다고 답하세요.

Context:
{context}
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    def retrieve_and_format(x):
        """검색 실행 및 포맷팅"""
        try:
            input_text = x["input"] if isinstance(x, dict) else x
            print(f"\n🔍 검색 쿼리: '{input_text}'")

            retrieved_docs = retriever.invoke(input_text)
            print(f"📊 검색 결과: {len(retrieved_docs) if retrieved_docs else 0}개")

            return format_docs(retrieved_docs)
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            return f"검색 중 오류 발생: {str(e)}"

    # 체인 구성: 검색 → 프롬프트 → LLM
    return (
        {
            "context": retrieve_and_format,
            "chat_history": lambda x: x["chat_history"],
            "input": lambda x: x["input"],
        }
        | prompt
        | llm
    )


def create_chain_without_kb(llm):
    """일반 대화용 체인 - KB 없이 LLM만 사용"""
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    return prompt | llm


def main():
    """메인 챗봇 실행 로직"""
    llm = get_llm()
    retriever = None
    
    # RAG 사용 여부 및 PDF 파일 경로 입력
    use_kb_input = input("💡 Knowledge Base (RAG)를 사용하시겠습니까? (y/n): ").strip().lower()
    
    if use_kb_input == 'y':
        pdf_path = input("📂 사용할 PDF 파일 경로를 입력하세요 (예: document.pdf): ").strip()
        retriever = get_retriever_from_pdf(pdf_path)
        if retriever:
            print("✅ Knowledge Base가 준비되었습니다. RAG 모드로 시작합니다.")
        else:
            print("⚠️ Knowledge Base 준비에 실패했습니다. 일반 대화 모드로 시작합니다.")
    else:
        print("ℹ️ 일반 대화 모드로 시작합니다.")

    # LangChain 형식의 대화 기록
    chat_history = []

    print("\n--- 챗봇 시작 --- (종료하려면 'exit' 또는 'quit' 입력)")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("👋 챗봇을 종료합니다.")
                break

            # 사용자 메시지를 기록에 추가
            chat_history.append(HumanMessage(content=user_input))

            # KB 사용 여부에 따라 체인 선택
            if retriever:
                chain = create_chain_with_kb(retriever, llm)
            else:
                chain = create_chain_without_kb(llm)

            print("AI: ", end="", flush=True)
            
            full_response = ""
            
            # 스트리밍 응답 처리
            # chain.stream 호출 시, 현재 입력을 제외한 이전 기록을 전달
            for chunk in chain.stream(
                {
                    "chat_history": chat_history[:-1], # 마지막 HumanMessage 제외
                    "input": user_input,
                }
            ):
                content = chunk.content
                print(content, end="", flush=True)
                full_response += content

            print() # 줄바꿈

            # AI 응답을 기록에 추가
            chat_history.append(AIMessage(content=full_response))

        except KeyboardInterrupt:
            print("\n👋 챗봇을 종료합니다.")
            break
        except Exception as e:
            error_msg = f"오류 발생: {str(e)}"
            print(f"\n❌ {error_msg}")
            # 오류 발생 시, AI 응답으로 오류 메시지 기록
            chat_history.append(AIMessage(content=error_msg))


if __name__ == "__main__":
    main()
