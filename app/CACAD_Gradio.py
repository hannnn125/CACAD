import gradio as gr
import yaml
from CACAD_32B import CACAD_32B

with open("configs/base_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# 봇 인스턴스 생성
bot = CACAD_32B(config=config)

def chat(user_message, history):
    """채팅 처리"""
    global current_session_id
    
    if current_session_id is None:
        return history
    
    if not user_message or not user_message.strip():
        return history
    
    # 사용자 답변 추가
    history.append([None, user_message])
    
    # 답변 처리
    bot.send_answer(current_session_id, user_message)
    
    # 다음 질문 가져오기
    result = bot.get_next_question(current_session_id)
    
    if result[0]:
        if result[0].get('completed', False):
            history.append(["모든 상담이 완료되었습니다.", None])
        else:
            question = result[0]['question']
            history.append([question, None])
    else:
        history.append(["오류가 발생했습니다.", None])
    
    return history


def reset():
    """초기화"""
    global current_session_id
    if current_session_id:
        bot.delete_session(current_session_id)
    return initialize_session()


# Gradio 인터페이스
with gr.Blocks(title="상담", theme=gr.themes.Soft()) as demo:
    with gr.Row():
        gr.Markdown("# 상담", elem_classes="center")
        init_btn = gr.Button("초기화", size="sm")
    
    chatbot = gr.Chatbot(
        height=600,
        show_label=False,
        avatar_images=(None, None)
    )
    
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="답변을 입력하세요...",
            show_label=False,
            scale=9,
            container=False
        )
        submit_btn = gr.Button("전송", scale=1)
    
    # 이벤트 핸들러
    init_btn.click(
        fn=reset,
        outputs=[chatbot]
    )
    
    submit_btn.click(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        outputs=[user_input]
    )
    
    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        outputs=[user_input]
    )
    
    # 페이지 로드 시 자동 초기화
    demo.load(
        fn=initialize_session,
        outputs=[chatbot]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
