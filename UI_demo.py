import streamlit as st
from chat import (
    search_demo, send_message_4o, 
    load_conversation, delete_conversation, 
    get_blob_url_with_sas, upload_to_blob_storage, upload_conversation_to_blob)
from openai import AzureOpenAI
import re
import index_doc

AZURE_OPENAI_SERVICE = "cog-kguqugfu5p2ki"
api_version = "2023-12-01-preview"
endpoint = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key="4657af893faf48e5bd81208d9f87f271"
    # azure_ad_token_provider,
)

# Initialize conversation
conversation = []
conversation_final = []

st.set_page_config(page_title="Deeeplabs Demo Chatbot", layout="wide")

st.title('Deeeplabs Demo Chatbot')

col1, col2, col3 = st.columns([1, 1, 1])
with col3:
    # Transcription_Mode = st.button('Transcription Mode', use_container_width=True)
    # if Transcription_Mode:
    st.markdown("""
    <a href="https://notebuddyview.z23.web.core.windows.net/" target="_blank">
    <button style='margin: 10px; padding: 10px; background-color: #FFFFFF; color: black; border: curve; cursor: pointer;'>Transcription Mode</button>
    </a>
    """, unsafe_allow_html=True)
    
    # Define the options for the dropdown
    options = ['GPT 4-o', 'GPT 4', 'GPT 3.5']

    # Create the dropdown
    selected_option = st.selectbox('Select model', options)

    # Display the selected option
    if selected_option == 'GPT 4-o':
        model = "gpt-4o"
    elif selected_option == 'GPT 4':
        model = "chat4"
    else:
        model = "chat16k"

logo_url = get_blob_url_with_sas('dl-logo-hamburger.png', "image")
st.sidebar.image(logo_url, width=180)
# with col1:
# Sidebar for system prompt
    # st.sidebar.header("Settings")
st.sidebar.markdown("<h1 style='text-align: left;'>System prompt</h1>", unsafe_allow_html=True)
system_prompt = st.sidebar.text_area(label = "", value = """You are a friendly and approachable conversational assistant who helps employees with questions and provide solutions on human resources (HR) issues employees face at work. You are a good listener and always emphatizing.
Always follow these response guidelines:
"First, start by introducing yourself to the employee as "Jenny" their friendly Company HR assistant who is happy to help them with any questions. Ask what is the employee’s name.
Next, always emphatize with the employee and answer the question. Follow up with a relevant question to continue the conversation.
Lastly, if the conversation is ending, summarise the key points and ask a follow up question to close the conversation.
At any time, feel free to add in some advice for the employee."
Organize your responses into clear paragraphs.
Use bullet points when needed.
Speak Naturally. 
Answer based on the information listed in the list of extracted sources below and provide suggestions or advice. If there isn't enough information below, say you don't know. If asking a clarifying question to the user would help, ask the question.""" , height=200)

# st.sidebar.markdown("<br>"*4, unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align: left;'>Upload File</h1>", unsafe_allow_html=True)
# Upload file
uploaded_file  = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "txt"], help="Upload a file to provide context")
# Check if a file was uploaded
if uploaded_file is not None:
    # Save the file to Azure Blob Storage
    file_url = upload_to_blob_storage(uploaded_file)
    # file_name = uploaded_file.name
    index_doc.run(uploaded_file)
    st.sidebar.write('File uploaded successfully!')

# Main layout
# Display conversation
st.header("Conversation")

delete_button = st.button('Clear chat')
if delete_button:
    st.session_state.messages = []
    try:
        st.session_state.messages = []
        delete_conversation("conversation.json")
        delete_conversation("history_json.json")
        st.write('Your chat has been deleted!')
    except:
        st.write('No chat to delete!')

# User input section
st.write(" ")
st.write(" ")

add_source = "\n\nProvide the relevant sourcepage in the end of the response. \
    Do not provide the irrelevant sourcepage. \
    The sourcepages always have the format .pdf. For example: 'Source: text1.pdf, text2.pdf'. \
    Do not provide the sourcepage if the question is generic"

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# User-provided prompt
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        history = load_conversation("history_json.json")['history']
        print(history)
    except:
        history = []

    summary_prompt_template = """Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation and the new question. Source names are not good search terms to include in the search query.

    Summary:
    {summary}

    Question:
    {question}

    Search query:
    """
        
    if len(history) > 0:
        completion = client.completions.create(
            model='davinci',
            prompt=summary_prompt_template.format(summary="\n".join(history), question=user_input),
            temperature=0.0,
            max_tokens=32,
            stop=["\n"])
        search = completion.choices[0].text
    else:
        search = user_input
    try:
        conversation = load_conversation("conversation.json")
    except:
        conversation = [
                {
                    "role": "system",
                    "content": system_prompt.replace('   ', '') + add_source
                }
            ]
    query = search_demo(search)['user_message']
    conversation.append({"role": "user", "content": query})
    response_0 = send_message_4o(conversation, model)

    # pattern = r'\b[\w\s-]+\.pdf-\d+'  
    # # Find all URLs in the text
    # resources_final = re.findall(pattern, response)
    if '.pdf' in response_0:
        resources_final = [source for source in search_demo(search)['source'] if '0.pdf' not in source]
    else:
        resources_final = []
    
    pattern = r"Source: [^\s]+\.pdf"
    response = re.sub(pattern, "", response_0).strip()
    # try:
    #     if resources_final[0] == '' or 'N/A' in resources_final[0]:
    #         resources_final = []
    #     else:
    #         resources_final = re.findall(pattern, response)
    # except:
    #     resources_final = []
    # response_1 = re.sub(pattern, "", response)
    response_final = response
    conversation[-1]['content'] = user_input
    conversation.append({"role": "assistant", "content": response_final})

    history.append("user: " + user_input)
    history.append("assistant: " + response_final)
    history_json = {"history": history}
    upload_conversation_to_blob("conversation.json", conversation)
    upload_conversation_to_blob("history_json.json", history_json)
    st.session_state.messages.append({"role": "assistant", "content": {"response": response, "resources": resources_final}})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        elif message["role"] == "assistant":
            st.write(message["content"]["response"].replace(")", "").replace("(", ""))
            resource_list = message["content"]["resources"]
            if len(resource_list) > 0:
                st.write("References:")
                for resource in resource_list:
                    resource_name = resource
                    reference_url = get_blob_url_with_sas(resource_name, "data-source")
                    st.write(f'[{resource_name}]({reference_url})')
