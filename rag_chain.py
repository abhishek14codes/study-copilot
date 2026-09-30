import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectorstore import vector_store
from langchain_core.runnables import RunnableParallel , RunnableLambda , RunnablePassthrough


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

retriever = vector_store.as_retriever(search_type="similarity" ,search_kwargs={"k":4})
query = input("enter the query")


prompt = PromptTemplate(
    template="You are a tutor and you will help anwer query : {query} while explaning concepts from : {context} , if you dont know the answer just say you dont know" ,
    input_variables=['query' , 'context']
)
parser = StrOutputParser()


def format_docs(retrevied_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrevied_docs)
    return context_text



parallel_chain = RunnableParallel({
    'context' : retriever | RunnableLambda(format_docs) ,
    'query' : RunnablePassthrough()
}
)

main_chain = parallel_chain | prompt | llm | parser

print(main_chain.invoke(query))
  


