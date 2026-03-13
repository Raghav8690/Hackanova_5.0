import traceback
import sys

try:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool
    print("OK: langchain_core")
except Exception:
    traceback.print_exc()
    print("---")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("OK: langchain_google_genai")
except Exception:
    traceback.print_exc()
    print("---")

try:
    from langchain_core.tools import tool
    print("OK: langchain_core.tools")
except Exception:
    traceback.print_exc()
    print("---")

try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    print("OK: langchain_core.prompts")
except Exception:
    traceback.print_exc()
    print("---")
