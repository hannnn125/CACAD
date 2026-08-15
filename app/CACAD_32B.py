import os
import json
import requests
from datetime import datetime
import uuid
import threading
import time 
from contextlib import contextmanager
from collections import defaultdict

class CACAD_32B:
    INITIAL_QUESTIONS = {
    "방임": "집에서 주로 너를 돌봐주는 어른은 누구야?",
    "성학대": "누가 몸을 만지거나 보여달라고 한 적 있어?",
    "신체학대": "주위 어른이 너를 때렸던 사람이 있어?",
    "정서학대": "주변 어른들 때문에 슬프거나 화가 난 적 있어?",
}   
    def __init__(self, config=None):
        self.config = config
        self.server_url = self.config['counseling'].get("server_url", "YOUR_VLLM_HOST")
        self.api_key = self.config['counseling'].get("api_key", "EMPTY")
        self.model_name = self.config['counseling'].get("model_name", "LGAI-EXAONE/EXAONE-4.0-32B-AWQ")
        self.abuse_types = ["방임", "정서학대", "신체학대", "성학대"]
        self.prompts={
            'base': self._load_prompt_file(os.path.join(self.config["paths"]["prompt_dir"], "counseling", "base_prompt.md")),
            'baseline': self._load_prompt_file(os.path.join(self.config["paths"]["prompt_dir"], "counseling", "baseline_prompt.md")),
            'token_prediction': self._load_prompt_file(os.path.join(self.config["paths"]["prompt_dir"], "counseling", "token_prediction_prompt.md")),
            'category': self._load_prompt_file(os.path.join(self.config["paths"]["prompt_dir"], "counseling", "category_prompt.md")),
            'follow': self._load_prompt_file(os.path.join(self.config["paths"]["prompt_dir"], "counseling", "follow_prompt.md"))
        }
        self.conversation_examples = self._gen_conversation_examples()
        #세션 관리
        self.sessions = {}
        self.session_lock = threading.Lock()

        # 대화 저장 관련 초기화
        self.conversation_save_dir = self.config["counseling"].get("conversation_save_dir", "./conversations")
        os.makedirs(self.conversation_save_dir, exist_ok=True)

        # NQCP 서버 URL 설정
        self.nqcp_url = self.config['counseling'].get("nqcp_url", "http://localhost:8001")
        self.offensive_url = self.config['counseling'].get("offensive_url", "http://localhost:8002")
        # 데이터 로드
        self._load_topQ_data()

    @contextmanager
    def _measure_latency(self, session, mode: str=None):
        start = time.time()
        yield
        latency = time.time() - start
        session['latency_log'].append({
            'type': 'question_generation',
            'mode': mode,
            'latency': latency,
            'timestamp': datetime.now().isoformat()
        })
        print(f"[질문 생성 Latency] {latency:.4f}초 ({mode})")

    def _append_assistant_turn(self, session, question: str):
        """assistant 응답 추가"""
        session['turn_number'] += 1
        session['current_cluster_history'].append(question)
        session['current_abuse_type_history'].append(question)

    def _build_response(self, session, question: str, mode: None, instruction: None, next_cluster: None):
        """응답 빌드"""
        return {
            'completed': False,
            'question': question,
            'current_step': session['current_abuse_type_index'] + 1,
            'total_steps': len(self.abuse_types),
            'instruction': instruction,
            'abuse_type': session['current_abuse_type'],
            'cluster': next_cluster,
            'mode': mode
        }, None

    def _call_vllm_api(self, messages):
        """vLLM API 직접 호출"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # API 키가 있으면 Authorization 헤더 추가
            if self.api_key and self.api_key.strip() and self.api_key != "EMPTY":
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 200,
                "top_p": 0.9,
                "chat_template_kwargs": {"enable_thinking": False}
            }
            
            url = f"{self.server_url}/chat/completions"
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            message = result['choices'][0]['message']
            reasoning_content = message.get('reasoning_content', '')
            content = message.get('content', '')
            
            if reasoning_content:
                return reasoning_content
            else:
                return content
                
        except requests.exceptions.RequestException as e:
            print(f"[debug] vLLM API 호출 오류: {e}")
            raise
        except KeyError as e:
            print(f"[debug] 응답 파싱 오류: {e}")
            raise
        
    def create_session(self):
        """새로운 세션 생성"""
        session_id = str(uuid.uuid4())
        
        with self.session_lock:
            self.sessions[session_id] = {
                'current_abuse_type_index': 0,
                'current_abuse_type': self.abuse_types[0],
                'current_abuse_type_history': [], #현재 학대유형 내 대화 히스토리
                'first_turn': True,
                'used_clusters': set(),
                'turn_number': 0,
                'completed': False,
                'current_cluster': None,
                'current_cluster_history': [],  # 현재 토픽의 대화만 저장
                'cluster_first_QApair': [],  # 각 클러스터의 첫 질문-답변 쌍 저장 (follow-up 제외)
                'latency_log': []
            }
            
        return session_id
    def _init_abuse_type_session(self, session_id):
        """새로운 학대 유형 세션 초기화"""
        session = self.sessions[session_id]
        
        # 현재 학대 유형 설정
        session['current_abuse_type'] = self.abuse_types[session['current_abuse_type_index']]
        
        # 세션 상태 초기화
        session['first_turn'] = True
        session['used_clusters'] = set()
        session['turn_number'] = 0
        session['current_cluster_history'] = []  # 토픽 대화 초기화
        session['follow_up_count'] = 0  # 추가 질문 횟수 초기화
        session['cluster_first_QApair'] = []  # 클러스터 히스토리 초기화 (새 학대 유형 시작 시)
        
        # 채팅 히스토리 초기화
        session['current_abuse_type_history'] = []
    def delete_session(self, session_id):
        """세션 삭제"""
        with self.session_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
        return False

    def _build_history_text_for_cluster_prediction(self, session_id):
        """클러스터 예측을 위한 대화 히스토리 텍스트 생성 (멀티턴 형식)"""
        session = self.sessions[session_id]
        
        # cluster_history에 저장된 모든 이전 토픽의 첫 질문-답변 사용
        cluster_history = session.get('cluster_first_QApair', [])
        if not cluster_history:
            encoder_input = "" # 첫 번째 토픽인 경우 빈 문자열
        else:
            # 멀티턴 형식: Q: 질문1\nA: 답변1\nQ: 질문2\nA: 답변2\n...
            lines = []
            for item in cluster_history:
                lines.append(f"Q: {item['question']}")
                lines.append(f"A: {item['answer']}")
            
            encoder_input = "\n".join(lines)
        return encoder_input
    def predict_next_clusters_probs(self, history_text, abuse_type):
        """클러스터 예측 결과와 확률값 반환 (NQCP 서버 호출)"""
        try:
            # 요청 데이터 검증 및 로깅
            if not abuse_type:
                raise ValueError(f"abuse_type이 None이거나 빈 값입니다: {abuse_type}")
            if history_text is None:
                history_text = ""
            
            request_data = {
                "history_text": history_text,
                "abuse_type": abuse_type
            }
            
            print(f"[debug] NQCP module request:")
            print(f"abuse_type: {abuse_type}")
            print(f"history_text 미리보기: {history_text[:100]}...")
            
            response = requests.post(
                f"{self.nqcp_url}/predict_cluster",
                json=request_data,
                timeout=30
            )
            
            # 400 오류인 경우 응답 본문도 출력
            if response.status_code == 400:
                try:
                    error_detail = response.json()
                    print(f"[debug] NQCP 서버 400 오류 상세: {error_detail}")
                except:
                    print(f"[debug] NQCP 서버 400 오류 응답: {response.text}")
            response.raise_for_status()
            result = response.json()
            # 응답 형식을 (cluster_id, probability) 튜플 리스트로 변환
            cluster_probs = [
                (item["cluster_id"], item["probability"])
                for item in result["cluster_probs"]
            ]
            return cluster_probs
        
        except requests.exceptions.RequestException as e:
            print(f"[debug] NQCP 서버 요청 실패: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"[debug] 오류 상세: {error_detail}")
                except:
                    print(f"[debug] 응답 본문: {e.response.text}")
            raise RuntimeError(f"[debug] NQCP 서버와 통신 중 오류 발생: {str(e)}")    
    def _get_next_cluster(self, session_id):
        """다음 클러스터 결정"""
        session = self.sessions[session_id]
        
        history_text = self._build_history_text_for_cluster_prediction(session_id)
        
        cluster_probs = self.predict_next_clusters_probs(history_text, session['current_abuse_type'])
        
        # print(f"클러스터 예측 결과 (상위 5개):")
        # for i, (cluster_id, prob) in enumerate(cluster_probs[:5]):
        #     print(f"  {i+1}. 클러스터 {cluster_id}: {prob:.4f}")
        
        # 사용하지 않은 클러스터 중 가장 확률이 높은 것 선택
        for cluster_id, cluster_prob in cluster_probs:
            if cluster_id not in session['used_clusters']:
                return cluster_id, cluster_prob
        
        # 예외처리: 사용하지 않은 클러스터가 없는 경우-None 반환
        print("모든 클러스터 사용됨")
        return None, cluster_probs
    
    def _get_conversation_examples(self, abuse_type):
        """학대 유형별 예시 대화 로드"""
        return self.conversation_examples.get(abuse_type, "")

    def _dialogue_from_data(self,data) -> str:
        dialogue = {}
        for item in data["list"]:
            category = item["항목"]
            audio = "\n".join(
                [f"Q: {x['text']}" if x["type"] == "Q" else f"A: {x['text']}" for x in item["audio"]]
            )
            dialogue[category] = audio
        return dialogue

    def _gen_conversation_examples(self):
        data_dir = os.path.join(self.config["paths"]["labeled_data"], "train")
        examples = defaultdict(list)
        for file in self.config["exclude_files"]:
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)
            dialogue = self._dialogue_from_data(data)
            for category, audio in dialogue.items():
                text = audio.strip()
                i = len(examples[category]) + 1
                examples[category].append(f"[예시대화{i}]\n{text}")
        example_texts = {
            category: "\n\n".join(blocks)
            for category, blocks in examples.items()
            }
        return example_texts
    
    def _load_topQ_data(self):
        """JSON 데이터 한 번만 로드하고 인덱싱"""
        cluster_path = self.config["counseling"].get("cluster_unique_question_path")
        self.cluster_index = {}  # 인덱싱 구조

        try:
            with open(cluster_path, "r", encoding="utf-8") as f:
                cluster_data = json.load(f)

            for abuse_type, items in cluster_data.items():
                self.cluster_index[abuse_type] = {}
                for item in items:
                    cluster = item.get('cluster')
                    text = item.get('text')
                    if cluster is None or not text:
                        continue
                    self.cluster_index[abuse_type].setdefault(cluster, []).append(text)
        
            print(f"클러스터 인덱싱 완료")
        except Exception as e:
            print(f"클러스터 데이터 로드 실패: {e}")
            self.cluster_index = {}
    def _load_category_definition(self, abuse_type, pred_cluster):
        """카테고리 정의 로드"""
        category_definition_file  = os.path.join(self.config['paths']['prompt_dir'],"cluster_details","cluster_definitions.json")
        with open(category_definition_file, "r", encoding="utf-8") as f:
            category_definition = json.load(f)
        return category_definition[abuse_type][str(pred_cluster)]
    
    def _get_category_information(self, abuse_type, pred_cluster, num_examples=None):
        """카테고리 정의 & 예시 질문 n 개 반환"""
        question_definition = self._load_category_definition(abuse_type, pred_cluster) 
        if not self.cluster_index or abuse_type not in self.cluster_index:
            return question_definition,"(참고 질문 없음)"
        
        questions = self.cluster_index[abuse_type].get(int(pred_cluster)) 
        if not questions or len(questions) < num_examples:
            return question_definition,"(참고 질문 없음)"
        sampled_questions = questions[:num_examples]
        question_example = "\n".join(f"{i}. {q}" for i, q in enumerate(sampled_questions, 1))
        return question_definition,question_example
    
    def _load_prompt_file(self, prompt_file):
        """프롬프트 파일 로드"""
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _load_prompt(self, abuse_type, prompt_type, pred_cluster=None, num_examples=None):
        """프롬프트 유형에 따라 시스템/예시 프롬프트를 조합해 반환."""
        base_prompt = self.prompts['base']

        match prompt_type:
            case "token_prediction":
                prompt = self.prompts['token_prediction'].format(abuse_type=abuse_type)
            case "baseline":
                example_conversations = self._get_conversation_examples(abuse_type)
                prompt = base_prompt +"\n" + self.prompts['baseline'].format(abuse_type=abuse_type, example_conversations=example_conversations)
            case "category":
                category_definition, category_Q_example = self._get_category_information(
                    abuse_type, pred_cluster, num_examples=2
                )
                example_conversations = self._get_conversation_examples(abuse_type)
                prompt = base_prompt + "\n\n" + self.prompts['category'].format(
                    abuse_type= abuse_type, 
                    category_definition=category_definition, 
                    category_example_questions=category_Q_example, 
                    example_conversations=example_conversations
                    )
            case "follow":
                example_conversations = self._get_conversation_examples(abuse_type)
                prompt = base_prompt +"\n" + self.prompts['follow'].format(abuse_type=abuse_type, example_conversations=example_conversations)
        return prompt
    

    def _check_offensive(self, question: str):
        """offensive 판별"""
        request_data = {
            "question": question
        }
        response = requests.post(
            f"{self.offensive_url}/predict_offensive",
            json=request_data,
            timeout=30
        )
        if response.status_code == 400: 
            try: 
                error_detail = response.json()
                print(f"[debug] Offensive 서버 400 오류 상세: {error_detail}")
            except: 
                print(f"[debug] Offensive 서버 400 오류 응답: {response.text}")
        response.raise_for_status()
        result = response.json()
        return result["is_offensive"]

    def _generate_until_safe(self, generate_once, fallback: str, max_retries: int = 2) -> str:
        for attempt in range(max_retries):
            question = generate_once()
            if not self._check_offensive(question):
                return question
            print(f"[debug] offensive 질문 재생성 ({attempt + 1}/{max_retries}): {question}")
        print("[debug] 재생성 한도 초과, fallback 사용")
        return fallback

    def get_next_question(self, session_id):
        """다음 질문 생성"""
        if session_id not in self.sessions:
            return None, "세션이 존재하지 않습니다."
        
        session = self.sessions[session_id]
        
        # 모든 학대 유형 완료 체크
        if session['current_abuse_type_index'] >= len(self.abuse_types):
            session['completed'] = True
            return {
                'completed': True,
                'question': None,
                'current_step': 4,
                'total_steps': 4,
                'instruction': "모든 상담이 완료되었습니다."
            }, None
        
        # 새로운 학대 유형 시작
        if session['first_turn']:
            self._init_abuse_type_session(session_id)
        
        # 이전 대화가 있는 경우 답변 충분성 분석
        if session['turn_number'] > 0 and len(session['current_abuse_type_history']) >= 2:
            # 추가 질문 횟수 확인 (최대 3회로 제한)
            follow_up_count = session.get('follow_up_count', 0)
            print(f"현재 추가 질문 횟수: {follow_up_count}")
            with self._measure_latency(session, 'answer_sufficiency'):
                sufficiency_result = self._analyze_answer_sufficiency(session_id)
            print(f"[debug] 답변 충분성 분석 결과: {sufficiency_result}")
            move_to_category=False

            if sufficiency_result == "|follow|" and follow_up_count < 3:
                # 추가 질문이 필요한 경우 (최대 3회 제한)
                print(f"추가 질문 모드: 답변이 불충분함 (횟수: {follow_up_count + 1}/3)")
                with self._measure_latency(session, 'follow_up_question'):
                    follow_up_question = self._generate_follow_up_question(session_id)

                if follow_up_question is not None:
                    session['follow_up_count'] = follow_up_count + 1
                    conversation_entry = {"role": "assistant", "content": follow_up_question}
                    self._append_assistant_turn(session, conversation_entry)
                    return self._build_response(
                        session, 
                        follow_up_question, 
                        'follow_up', "추가 질문을 통해 더 자세한 정보를 수집합니다.",
                        session.get('current_cluster', None))
                else:
                    print("[debug] follow 재생성 실패, 다음 클러스터로 이동")
                    move_to_category = True
            
            # |cluster| 또는 follow_up 생성 결과 offensive 2회 초과 또는 follow_up 3회 초과 시 다음 주제로 이동
            if sufficiency_result == "|cluster|" or follow_up_count >= 3 or move_to_category:
                print(f"다음 클러스터로 이동")
                current_cluster_history = session.get('current_cluster_history', [])
                if len(current_cluster_history) >= 2:
                    session['cluster_first_QApair'].append({
                        'question': current_cluster_history[0]['content'],
                        'answer': current_cluster_history[1]['content']
                    })
                
                session['current_cluster_history'] = []
                session['follow_up_count'] = 0

        
        if session['turn_number'] == 0:
            # 첫 턴인 경우 고정 질문 사용 (클러스터 예측 불필요)
            question=self.INITIAL_QUESTIONS[session['current_abuse_type']]
            next_cluster = None
            session['current_cluster'] = None
        else:
            next_cluster, cluster_probs = self._get_next_cluster(session_id)
            if next_cluster is None or next_cluster == -2:
                # 현재 학대 유형 완료, 다음 유형으로 이동
                session['current_abuse_type_index'] += 1
                session['first_turn'] = True
                
                if session['current_abuse_type_index'] < len(self.abuse_types):
                    return self.get_next_question(session_id)
                else:
                    session['completed'] = True
                    return {
                        'completed': True,
                        'question': None,
                        'current_step': 4,
                        'total_steps': 4,
                        'instruction': "모든 상담이 완료되었습니다."
                    }, None
            
            # 종료가 아닌 경우, 클러스터 사용 처리
            session['used_clusters'].add(next_cluster)
            
            # 새로운 클러스터로 이동할 때 상태 초기화
            session['follow_up_count'] = 0
            session['current_cluster_history'] = []
            print(f"새로운 클러스터 {next_cluster}로 이동, follow_up_count 초기화")
            
            with self._measure_latency(session, 'cluster_question'):
                question = self._generate_category_question(session_id, next_cluster)
            session['current_cluster'] = next_cluster
        
        session['first_turn'] = False
        conversation_entry = {"role": "assistant", "content": question}
        self._append_assistant_turn(session, conversation_entry)
        return self._build_response(session, question, 'cluster', instruction = None, next_cluster=next_cluster)

    def send_answer(self, session_id, answer):
        """사용자 답변 처리"""
        if session_id not in self.sessions:
            return None, "세션이 존재하지 않습니다."
        
        session = self.sessions[session_id]
        
        # 답변 검증
        if answer.strip().lower() == '|end|':
            session['completed'] = True
            return "세션이 종료되었습니다.", None
        
        conversation_entry = {"role": "user", "content": answer}
        session['current_abuse_type_history'].append(conversation_entry)
        session['current_cluster_history'].append(conversation_entry)
        
        # 최대 턴 수 확인
        max_turns = self.config['counseling'].get("max_turns", 20)
        if session['turn_number'] >= max_turns:
            # 현재 학대 유형 완료, 다음 유형으로 이동
            session['current_abuse_type_index'] += 1
            session['first_turn'] = True
            
            # 다음 학대 유형을 위해 상태 초기화
            session['current_abuse_type_history'] = []
            session['used_clusters'] = set()
            session['turn_number'] = 0
        
        return "답변이 처리되었습니다.", None
    
    def _analyze_answer_sufficiency(self, session_id):
        """아이의 답변이 충분한지 분석하여 |cluster| 또는 |follow| 반환"""
        session = self.sessions[session_id]
        
        # 답변 충분성 판단을 위한 프롬프트 생성
        abuse_type = session['current_abuse_type']
        analysis_prompt = self._load_prompt(abuse_type, "token_prediction")
        current_cluster_history = session.get('current_cluster_history', [])
    
        analysis_context = [{"role": "system", "content": analysis_prompt}] + current_cluster_history
        
        try:
            result = self._call_vllm_api(analysis_context)
            if "|cluster|" in result:
                print(f"|cluster| 토큰 발견")
                return "|cluster|"
            elif "|follow|" in result:
                print(f"|follow| 토큰 발견")
                return "|follow|"
            else:
                print(f"토큰을 찾을 수 없음: '{result}', 기본값 |cluster|로 설정")
                return "|cluster|"
                
        except Exception as e:
            print(f"답변 충분성 분석 오류: {e}")
            return "|cluster|"

    def _generate_category_question(self, session_id, pred_cluster):
        """클러스터 정보를 내부적으로 활용하여 질문 생성"""
        session = self.sessions[session_id]
        abuse_type = session['current_abuse_type']
        category_prompt = self._load_prompt(abuse_type, "category", pred_cluster)
        # 질문 생성 컨텍스트 구성 (system + 전체 대화 히스토리)
        context = [{"role": "system", "content": category_prompt}] + session['current_abuse_type_history']
        def generate_once():
            try:
                response = self._call_vllm_api(context)
                
                # 특수 토큰 제거 및 정리
                question = response.strip()
                question = question.replace("[|endofturn|]", "").replace("[|assistant|]", "").strip()
                category_question = question.split("\n")[0].strip()
                return category_question 
                
            except Exception as e:
                print(f"[debug] 질문 생성 오류: {e}")
                return "안녕? 오늘 어떤 일이 있었어?"
        category_question = self._generate_until_safe(generate_once, fallback=None)

        if category_question is None:
            _, category_Q_example = self._get_category_information(
                    abuse_type, pred_cluster, num_examples=1
                )
            category_Q_example = category_Q_example.removeprefix("1. ").strip()
            return category_Q_example
        else:
            return category_question

    def _generate_follow_up_question(self, session_id):
        """현재 질문에 대한 추가 질문 생성"""
        session = self.sessions[session_id]
        abuse_type = session['current_abuse_type']
        follow_up_prompt = self._load_prompt(abuse_type, "follow")
        follow_up_context = [{"role": "system", "content": follow_up_prompt}] + session['current_abuse_type_history']

        def generate_once():
            try: 
                follow_up_question = self._call_vllm_api(follow_up_context)
                follow_up_question = follow_up_question.split("\n")[0].strip()
                return follow_up_question 
            except Exception as e:
                print(f"[debug] 추가 질문 생성 오류: {e}")
                return "조금 더 자세히 말해줄 수 있어?"
        follow_up_question =self._generate_until_safe(generate_once, fallback=None)
        if follow_up_question is None:
            return None
        else:
            return follow_up_question