import os.path

import pandas as pd
from openai import OpenAI  # 假设使用OpenAI API，你也可以替换成其他大模型API
from zhipuai import ZhipuAI # 百度智谱AI
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

# input_file = "./data/test_anxiety_test.csv"  # 替换为你的CSV文件路径
# 千问支持的模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
# openai gpt系列："gpt-3.5-turbo"   "gpt-4o"
# GLM系列："glm-4-plus"
# 阿里-大模型平台百炼（里面集成了很多模型，包括开源的<不想自己部署的可直接调用>）："qwen-plus"  "qwen2.5-72b-instruct"
model_name = "deepseek-chat" # "gpt-3.5-turbo" "gpt-4o" "glm-4-plus" "deepseek-chat" "qwen-plus" "qwen2.5-72b-instruct"
disorders = ["anxiety","bipolar","depression","post-traumatic stress disorder","suicide ideation"]
# 路径名称根据实际存放文件的位置修改即可
input_files = ["./data/test_anxiety.csv", "./data/test_bipolar.csv", "./data/test_depression.csv", "./data/test_ptsd.csv", "./data/test_self_harm.csv"]
risk_levels = {'crisis': 3, 'red': 2, 'amber': 1, 'green': 0}

# def calculate_micro_f1(gold_labels, pred_labels):
#     """
#     计算micro F1值
#     Args:
#         gold_labels: 真实标签列表/数组 (包含0或1)
#         pred_labels: 预测标签列表/数组 (包含0或1)
#     Returns:
#         micro_f1: micro F1值
#         additional_metrics: 包含其他指标的字典(TP,FP,FN,TN,precision,recall)
#     """
#     # 确保输入长度相同
#     if len(gold_labels) != len(pred_labels):
#         raise ValueError("gold_labels和pred_labels长度必须相同")
#
#     # 初始化计数器
#     TP = 0  # True Positive
#     FP = 0  # False Positive
#     FN = 0  # False Negative
#     TN = 0  # True Negative
#
#     # 计算TP, FP, FN, TN
#     for gold, pred in zip(gold_labels, pred_labels):
#         if gold == 1 and pred == 1:
#             TP += 1
#         elif gold == 0 and pred == 1:
#             FP += 1
#         elif gold == 1 and pred == 0:
#             FN += 1
#         elif gold == 0 and pred == 0:
#             TN += 1
#
#     # 计算precision和recall
#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#
#     # 计算micro F1
#     micro_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#
#     # 构建包含其他指标的字典
#     additional_metrics = {
#         'TP': TP,
#         'FP': FP,
#         'FN': FN,
#         'TN': TN
#     }
#     print(f'additional_metrics={additional_metrics}')
#
#     return precision, recall, micro_f1


def calculate_f1(gold_labels, pred_labels, average='macro'):
    '''
    计算 Precision, Recall, F1
    gold_labels：真实标签
    pred_labels：预测标签
    average:
        'binary': 用于二分类问题（默认正类为 1）。
        'micro': 全局计算 Precision、Recall 和 F1。
        'macro': 每类的 P、R、F1 平均（不考虑类别不平衡）。
        'weighted': 每类的 P、R、F1 按支持样本数加权（考虑类别不平衡）。
        None: 返回每个类别的 P、R、F1（适用于多分类）。
    '''
    precision, recall, f1, _ = precision_recall_fscore_support(gold_labels, pred_labels, average=average)
    acc = accuracy_score(gold_labels, pred_labels)
    # print(f"Precision (P): {precision}")
    # print(f"Recall (R): {recall}")
    # print(f"F1 Score: {f1}")
    return acc,precision, recall, f1


def predict_classid(disordername, post):
    """
    使用大模型预测文本是否为郁抑症帖子
    Args:
        disordername: 疾病名称
        post: 需要预测的文本内容
    Returns:
        int: 1表示是郁抑症帖子，0表示不是
    """
    try:
        # 设置你的API密钥
        if 'glm-4' in model_name:
            # print('正在使用glm-4大模型API')
            client = ZhipuAI(api_key="Your glm-4 APIKey")  # 填写您自己的APIKey
        elif 'qwen' in model_name:
            # print('正在使用qwen大模型API')
            client = OpenAI(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                # api_key=os.getenv("DASHSCOPE_API_KEY"),    # 通过配置环境变量设置api_key
                api_key="Your qwen APIKey",  # 直接使用api_key
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif 'deepseek' in model_name:
            client = OpenAI(
                api_key="Your deepseek APIKey",
                base_url="https://api.deepseek.com")
        else:  # gpt-3.5-Turbo  gpt-4o
            client = OpenAI(
                # openai系列的sdk，包括langchain，都需要这个/v1的后缀
                base_url="https://api.openai.com/v1",   # https://api.openai-proxy.org/v1
                api_key="Your GPT APIKey",
            )

        # 构造prompt
        # prompt = f"""请根据以下帖子内容，判断该发帖人是否存在抑郁症？
        # 请仅用"是"或"否"回答。
        # 帖子内容：{post}
        # """
        if disordername == 'suicide ideation':
            prompt = f"""Give a post content, Please judge the suicide risk level of the poster?
                        There are four levels of suicide risk, as follows:
                        Green: No intervention required, indicating that the poster exhibits no suicidal ideation and thus no risk of suicide or self-harm.
                        Amber: Intervention required but not urgent, indicating that the poster exhibits a moderate risk of suicide or self-harm.
                        Red: Intervention required and urgent, indicating that the poster exhibits a high risk of suicide or self-harm.
                        Crisis: Intervention required and extremely urgent, indicating that the poster is at an imminent and severe risk of suicide or self-harm.
                        Requirement: Please answer only with "Green", "Amber", "Red" or "Crisis".
                        Here are some examples:
                        Post content:  going back to america in less than a week and coming down with a serious case of the feels 
                        Label: Green
                        Post content:   @ blithe thank you for your awesome advice - i feel much better. i had one of those mood swings where you feel like nothing can be achieved and all the self esteem drops and stuff. i am over it - i felt like i overreacted but i guess it got to do with the mood. thank you for your suggestions and advice , the websites were useful , thank you. smiley happy 
                        Label: Amber
                        Post content: my girlfriend left me yesterday.... she said that half of her loves me and half of her keeps loosing love everytime i do so somthing. she says i keep shocking her into loving me less... i love her so much and she is the only thing i have in mh life.... i have no friends too talk to or anyone other than her and she wont speak to me... i cant sleep and if i do i dream about me and her together and i wake up... i dont want to lose her... she is the one i want to spend the rest of my life with and i want to fix things and not mess it up
                        Label: Red
                        Post content: redhead: aaaaarrrrgggg getting really sick of this hypomania. when will this end. getting so agitated , bored. and its raining so i ca n't even get out for a walk to calm down. what to do. its starting to make me feel unsafe...smiley happy sorry for being so negative again
                        Label: Crisis
                                                                       
                        Post content: {post}
                        Label:
                        """
        else:
            if disordername == 'anxiety':
                prompt = f"""Given the content of a post, please determine whether the poster exhibits signs of {disordername}. Respond with either 'Yes' or 'No' only.
                             Here are some examples.
                             Post content: i have severe anxiety and depression and have made a recovery through therapy, medication and hard work over the past few years. i spent close to 10 years in a deep anxiety ridden depression, suffering in silence until my life fell apart and i hit rock bottom. i used to never talk about it, and was ashamed of and embarrassed by it. but now after therapy i find myself talking about it maybe too openly, and have found myself over sharing in other areas too. i will catch myself after and wish i hadn't said what i did, and hope it wasn't awkward for the person i am speaking to. has anyone else experienced this? what did you do to balance it out?
                             Label: Yes
                             Post content: not that i had much to begin with, but lately my faith in any treatment regimen working has been at an all-time low. to give a rough sketch recapitulating these past few months: got up to the target dose on lamictal (200-250mg), which was controlling some of my symptoms alright, made me functional enough to work at 50% mental capacity, but at least consistently so. however the cognitive side-effects and utter emotional dullness was simply not worth it to me. yes, i'd be capable of doing a little bit more and my energy was slightly more even-keeled throughout the day, but what's the point of having more energy if you don't want to do anything? if even music lost its emotional impact? so lamictal, you were not the worst, but you're just not worth it. i'll quit my job before resigning to a life of constant emotional flatness (not to be confused with euthymia, this is closer to a listless depression). got into a rebound relationship with lithium, which i must say is impressive for its lack of side-effects, and we had a short run where it controlled my symptoms which of course was short-lived and didn't last. suddenly, extreme fatigue and psychomotor retardation everyday. performance deterioration at work is starting to raise red flags with my coworkers, total carelessness and lack of attention to details seeping into my work which is hugely embarrassing and a great hit to my ego/self-esteem. to try and combat some of the fatigue, i took out remeron, a drug that i was on for years, cold-turkey (yes, i know, but i discussed it with my pdoc sort of first) in hopes that it would help. triggered hypomania, followed by rapid-cycling which was better than before, because at least i'd get some hours in the week where i'm functional. now i'm back to vanilla bp ii depression, and reintroducing remeron in any dosage doesn't seem to quell the fire, and in fact only made things worse. if i can't get out of this episode quickly, going to have to go on disability in a job i just started a few months ago, which will raise a lot of questions in my tiny 4-people workplace. but i don't even give a shit about that. whenever i talk to normals about these problems they focus on job, schooling, relationships like i could give a shit. i've been unemployed, sick, and socially ostracized before, that doesn't faze me, i'll survive somehow. i'm just sick of being _disabled_. i'm sick of spending 90% of my life in bed or on the couch contemplating the logistics of making a bowl of cereal. i. can't. do. it. anymore. it doesn't end, there's no respite, nothing ever works...all these condescending doctors talking with a false sense of authority about a subject they know next to nothing about; i can't even look them in the eye anymore. do they realize that in between prescription and dosage changes all i do is wait in purgatory-like experience lying prostrate on my floor? do they not understand how silly it is to ask if i have hobbies or what i did this past vacation?
                             Label: No
                             Post content: eight years ago today i had had enough and sat on the sea wall at midnight about to throw myself in. single, victim of dv, no full time job. i had terrible flashbacks, panic attacks, couldn't sleep. now i'm still anxious but medication has helped enormously. i have a permanent full time job i love, a partner i love and i got to see my boys become fine young men. see your doctor. you can get through this x
                             Label: Yes
                             Post content: i should look for a job but procrastination has taken a toll on me. my near future is uncertain and i feel unable to take responsible decisions for myself. i have to leave my flat in the end of the month. i will probably end up living at my parents again for a few months until i find something to do with my life. in november, it will be one year since i haven't been intimate with a guy and it makes me feel like i am loosing my femininity. i don't know who i am, don't know what i want, barely know how to get the things i need.
                             Label: No
                             Post content: i see it all the time. people saying to always listen to your instincts or always follow your gut. well that doesn’t always work for me. for example i have a deep fear of financial isn’t ability and losing my job and i constantly get “gut feelings” that my boss is mad at me or going to fire me soon. i get gut feelings that my plane is going to crash, that i have cancer, etc. if i listened to my gut i would never leave the house except to go to my many doctors appts to make sure i don’t have this or that disease. i do listen to my gut when i’m alone in a dark sketchy area of course, or dealing with someone who has shown themselves to be aggressive or dangerous or threatening in some way (like my abusive dad). but other than that i have to learn to live with it and not let it dictate what i do
                             Label: Yes
                             Post content: cw; death, violence last sunday i found the dead bodies of my two best friends folded into trash bags in their shed. i can’t unsee it. i can’t unsmell it. i can’t sleep. every time i close my eyes i see them in the bags. i can’t look at garbage bags anymore. i can’t look at sheds anymore. i drove my brother home yesterday and it was garbage day. i swear to god i could smell them. i don’t know what to do. i can’t function. i can’t get through a night at work. i can’t eat. i can’t even pull myself out of bed most of the time. i can’t do anything. i keep putting on a strong front for everyone else who is grieving and they keep saying that they’re here for me but they don’t /get/ it, you know? they are grieving, but they didn’t find them. they don’t get it. they don’t understand. and they keep saying super insensitive things about it. i don’t know what i am supposed to do anymore. i’m hanging by a thread here. they were my only friends. it was so bad. they were unidentifiable. it was so bad. there was so much blood. i don’t know how to stop seeing it. i can’t stop. it won’t go away what am i supposed to do
                             Label: No
                             Post content: you are enough regardless of whether you face your fears or avoid them. you are enough regardless of how many chores you completed or if you stayed in bed and watched tv. you are enough. you are good enough. you are acceptable. you are loveable. you are worthy. you deserve love and belonging. today i’m going to ignore the voice that says otherwise and listen to this voice. i’m going to be vulnerable and believe i am enough. will you do that with me? you are enough.
                             Label: Yes
                             Post content: fortunately i also forgot to turn the oven on, so i have nasty rotten potatoes instead of a house fire. win? this is exactly why i bought a toaster oven with a timed shut off, and a kettle that turns off when boiled, and an induction cooktop that dosen't heat up if there's no pan on it...sometimes living with my brain feels like having a roommate who wants me to die in a fire.
                             Label: No
                             Post content: this has been happening at work a lot lately, (in fact, right now) and it's getting harder to control it. earlier this evening, i made a huge mistake. like, hundreds of dollars that are coming out of my own paycheck huge. when i was calling my boss to explain the situation, he though that was a good opportunity to mention that i've been getting several other "complaints" from customers, as well. which, of course, skyrocketed the anxiety and made me cry harder. i'm just so sick of putting all my effort and trying so hard at work, only for it to be overshadowed by my mistakes and complaints. but i have customers who compliment me all the time, i have people who bring their friends and family to our store specifically because of me, and my bosses completely ignore that and it just makes me feel like shit. i've already had 2 major anxiety attacks in front of my boss, and i'm afraid a third will cross the line and he'll fire me. i don't have health insurance and got denied for medicaid so i can't see a therapist. i've been stealing my dad's xanax. :( i just need to figure out how to control my anxiety at work, and how to not feel worthless and like i want to die when i make mistakes. edit: medicaid, not medicare
                             Label: Yes
                             Post content: hi reddit! my name is amanda morris and i am a professor of chemistry at virginia tech. i serve as an acs expert in the field of sustainable energy this is my second ama on this topic – you can see my biography and that session’s discussion via some of the questions people had last year concerned: how does artificial photosynthesis work; how do the various semiconductor materials used in solar energy work; how does the electrical grid design affect solar energy potentials; how does energy storage technology play a role with solar uptake? i’m happy to answer more about these topics and any others that come to mind. ask me anything about solar energy! i will return at 11 am et (8am pt, 3pm utc) to answer your questions, ask me anything! (10:50 est) hey y'all! i am signed on and replying! i am pumped to see so many comments already! (1:00 est) great time answering your questions! the biggest themes were when is solar cost effective for me, solar cell recycling and life cycle analysis, and tesla's solar relevant technologies. there are calculators to consider the first point - is just one example. first solar is now recycling solar cell modules for large installations - great news! and, tesla continues to push the envelope, but the cost effectiveness of such technoogies isn't proven yet. thanks all! sorry, i didn't get to most of your questions! until next time! -acs edited adding utc and fixing punctuation
                             Label: No

                             Post content: {post}
                             Label:
                             """
            elif disordername == 'bipolar':
                prompt = f"""Given the content of a post, please determine whether the poster exhibits signs of {disordername}. Respond with either 'Yes' or 'No' only.
                             Here are some examples.
                             Post content: before you point out the obvious, i'm well aware this is a pretty pathetic problem to have. i don't expect anyone to take me seriously because it's so childish and ridiculous. but i've always had this deep rooted problem from my trauma where i've relied on my lying to cope. i don't mean the kind of lying for feeding my ego, pride, or unfounded bragging. i had to lie to protect myself from intense violence in my upbringing. my abuser's also forced me to lie to protect them. it almost happens as a knee jerk reaction now. i do it when i read into anything with a perceived threat that may not actually be there. i know it damages trust, but i struggle to care about that when i can't trust anyone myself. with that said, i hate hurting people and i'm deeply ashamed of this problem. about a year ago i decided if i couldn't break this habit i was going to kill myself because i couldn't live like this anymore. i've gotten this far and managed to break the habit to some extent. i had such a fucked up childhood i have to fabricate having a normal one just to even talk to others. i have to lie to myself and pretend i'm a person who never went through any of you. during mania i almost seem to get such a high from lying to myself like this. i genuinely wish i had a good childhood so it feels good to lie so that i can pretend just for a moment that the pain never happened. it's pitiful and stupid, i need to learn to actually be vulnerable and stop this. i seem like such an honest person that people have been shocked when i've come out to them as a compulsive liar. it was such a second nature for me growing up that i wouldn't even recognized when i lied sometimes. i didn't want to believe my parents were abusive, so it was easy to believe my own lies. i feel like some kind of stupid child having this problem and i wonder if it indicates my childhood hindered my development somehow. i did have a head injury when i was 11 and i suspect it made this worse. i refuse to make excuses though, because regardless of the explanation it's not a justification and i need to stop. i'm sick of depriving others of the truth because it's what they want to hear. i don't know if i just do this because i'm a people pleaser or a coward, but i need to change.
                             Label: Yes
                             Post content: i feel like i have been struggling with anxiety and depression for about half my life. coming from a family that doesn't see mental health as an importance, i've never really looked for help. i'm 24 now and i finally think its time to seek for help. i basically jut called the number on the back of my insurance card and they set me up with a therapist and psychiatrist. i typically avoid most doctors unless its absolutely necessary, but i've reached my limit here. i'm just so nervous/stressed/anxious about the upcoming appointment. my biggest fear with doctors in general is not being taken seriously and them understanding what i am trying to say. when i get put on the spot i tend to not explain myself very well. or what if i say the wrong thing because i couldn't explain myself and they end up thinking i'm a lot worse than i really am. i don't drink or do any other drugs besides weed, and i'm afraid telling them will change my treatment or any future medications. people with experience please give me some insight/reassurance. thank you!
                             Label: No
                             Post content: original post: so i talked briefly about the changes in my personality caused by the adhd diagnosis and how it related to bipolar. someone asked me to report back in a month to ensure it wasn't mania. i can tell you it's not. the strattera has worked a miracle. no noise in my head, i can understand conversations. i can focus. i can read and watch movies. i can follow a conversation. i sleep well. in short, i have found stability, and it's wonderful. hope all of you can find the same thing.
                             Label: Yes
                             Post content: i know that this is probably pathetic, but my depression got very bad and i didn't shower for months, now i'm back on track, going to try and keep going in every day now. sorry if i bothered you.
                             Label: No
                             Post content: so i spoke with my psychiatrist and he is taking me off seroquel and putting me on lithium. i’m so scared but i’m so relieved to be trying this. i really hope it helps me.
                             Label: Yes
                             Post content: i find that many people struggle with emotionally invalidating loved ones, yet the resources for a layperson to improve and address this feel lacking. specifically, my so has become rather bad at handling any expression of negative emotion, towards himself or otherwise, and my ptsd diagnosis seems to breed resentment in him at the idea it is not "all in my head" (like he hates that he cant say i am being too sensitive). he has openly explained that he struggles with empathy, that he would like to be better at being in touch with his feelings, but he also uses reason and logic to deflect any perspectives that dont match his own. i have been receiving therapy for some time and am at a very good place. however, hearing i cant/shouldnt feel a certain way does actually trigger me. he tends to exacerbate this and i dont know what to do. any insight...?
                             Label: No
                             Post content: i've mostly had psychiatrists. i've never been to therapy, outside of counselors you have to go to to refer you to psychiatrists. in fact, that was just what this was. a hoop to jump through to see a psychiatrist. when she asked me what my i liked to do, i didn't have an answer. she didn't accept school as a valid answer. she wouldn't let it go either... and 5-10 minutes later my best answer was still a shrug. i have no passion. where did it go? did it ever exist? any of you guys have this?
                             Label: Yes
                             Post content: idk why i seem to be unable to break the pattern of avoidance/depression/making amends/loosing trust and having to go above beyond to prove myself. inattentive type, im in therapy, emdr and meds. especially returning calls or emails. i have almost ptsd about opening my inbox and seeing someone mad at me or waiting on something i didnt do. then i avoid them and go incommunicado (which i know is literally the worst thing i could do). its almost like i cannot make myself admit i got behind to someone, and then i sit in my shame haunted by their emails/calls and it literally becomes a mountain from a molehill. anyone else struggle with this and how are you overcoming. i am working as hard as i can on this issue. staying ahead of my schedule and usig reminders/ planner to remembering things is helping, but theres always something i forget to do sometimes. i need a painless way to quickly admit i forgot and get the thing done, but my brain literally wont do what i'm screaming at to do.
                             Label: No
                             Post content: i just had a minor freakout over something i said, basically i told my boyfriend i wasn't up for d&amp;d tonight (which he was okay with) but i feel like i ruin everything and i hid myself away in the bathroom. i ended up getting really angry at myself during this and threw my phone at the wall, which prompted my mother to come upstairs and ask what was going on. i blew up at her and told her "everything's fine, just please leave me alone" (which was obviously yelled, not calmly spoken). i then moved to my room and started sobbing which prompted my boyfriend to come in and try to comfort me. we talked it out and i started to calm down, but when he was rubbing my back it felt like my skin was burning so i pulled away. i don't really know how to describe it other than an adverse tactile reaction to touch. i notice it a lot after i have a breakdown like this. tl;dr had mini-breakdown, lashed out and regretted it, talked it out with boyfriend and when he tried to physically comfort me it felt like my skin was burning.
                             Label: Yes
                             Post content: hi reddit! my name is amanda morris and i am a professor of chemistry at virginia tech. i serve as an acs expert in the field of sustainable energy this is my second ama on this topic – you can see my biography and that session’s discussion via some of the questions people had last year concerned: how does artificial photosynthesis work; how do the various semiconductor materials used in solar energy work; how does the electrical grid design affect solar energy potentials; how does energy storage technology play a role with solar uptake? i’m happy to answer more about these topics and any others that come to mind. ask me anything about solar energy! i will return at 11 am et (8am pt, 3pm utc) to answer your questions, ask me anything! (10:50 est) hey y'all! i am signed on and replying! i am pumped to see so many comments already! (1:00 est) great time answering your questions! the biggest themes were when is solar cost effective for me, solar cell recycling and life cycle analysis, and tesla's solar relevant technologies. there are calculators to consider the first point - is just one example. first solar is now recycling solar cell modules for large installations - great news! and, tesla continues to push the envelope, but the cost effectiveness of such technoogies isn't proven yet. thanks all! sorry, i didn't get to most of your questions! until next time! -acs edited adding utc and fixing punctuation
                             Label: No

                             Post content: {post}
                             Label:
                             """
            elif disordername == 'depression':
                prompt = f"""Given the content of a post, please determine whether the poster exhibits signs of {disordername}. Respond with either 'Yes' or 'No' only.
                             Here are some examples.
                             Post content: i feel so lost. i don't care about my future, i don't care about god, i don't care about family, i don't care about friends, i don't even care about myself. i have nothing to keep me going. edit: i didn't expect that many replies on my thread. so, thanks for everyone who shared how they feel and for the people who tried to help. i hope that all of you might find happiness one day
                             Label: Yes
                             Post content: went to group therapy today and acted like a complete ass (again), it was unintentional of course but totally monopolising the conversation and interrupting people and way oversharing. lucky it’s a ‘safe space’ but supposed to start a depression based group next week and we’re so not depressed right now i really don’t want our presence to cause someone harm more sadness or for them to stop coming because we are like this. omg i hate it!!!! but short of cutting out our tongue i am at a complete loss as how to get us to stfu. we’re embarrassing everyone in range. why can’t we just stop freaking talking??!!! augghhhhhh!!!! think we’ll just have to not go anymore :-( it’s not fair on everyone else.
                             Label: No
                             Post content: i’m 21 and i’ve had mental health issues for about 10/11 years now. during this time i’ve dealt with low mood, anxiety, ocd along with other things, including social anxiety. i rarely left my house unless i had to (and even then i didn’t sometimes), and that’s done nothing but worsen my social anxiety. i can’t eat around people, i can’t be around people for too long, and i’ve noticed my ability to speak has got increasingly worse and i now have a bit of a stutter that i can tell is getting worse everyday. despite the fact that i’ve always been nice to people and helped them out when they needed it, i feel very alone. i have a couple of friends, but every time i talk to them about anything i feel like i’m bothering them. this makes me feel even worse because i then quickly spiral into “no one really cares about me” sort of thoughts. the hardest part is, i can’t separate which parts of my personality are me, and which parts are my depression. i just feel so low all of the time. i’ve finally taken the plunge to go to a therapist, but i’ve been on the waiting list until september and i’m probably going to be waiting another couple of months. i just don’t know what to do. i don’t know why i wrote this or what i felt posting this would do, but i just wanted to put it down somewhere. thanks if you made it this far &lt;3
                             Label: Yes
                             Post content: i don’t want to talk about politics in an anxiety subreddit, but i can’t think straight anymore. everyday the tension between the us and north korea seems to be getting worse and worse. i can’t help but to go on news websites and check how the situation is unfolding, and it’s not looking good. i try to calm myself down by watching some videos on youtube, it does help me for a while, but then some late night show shit gets recommended to me. i can’t help but click it. they’re on the topic about north korea and their threats and it boggles me that they joke around this much about the possibility of a war, making remarks such as, “we’re all gonna die.” then the audience laughs their asses off, it really fucks me up. all this anxiety is making me unmotivated to do anything i like, knowing that i’ll think about nuclear war, even though war has nothing to do with the activity i’m doing. i’m a sophomore in high school so i’ve never really experienced anything like this before. i know i can’t do anything about it, it’s not in my control, but i just want answers, are we fucked?
                             Label: No
                             Post content: i love the idea of it because it removes so much stress from my life. for example: i am failing in school and will probably need to go to summer school but i can just kill myself making any work i do useless. it's like a get out of jail free card but instead of jail it's a get out of hell free card. fuck living.
                             Label: Yes
                             Post content: my trauma doesn’t even include the police, or even the kind of violence that we’re seeing committed on the streets of the us right now. i experienced violence, sure, but not to the extent of what i’m seeing every day on reddit. nevertheless i’m feeling really terrible lately. every time i see a video of an officer hurting someone, whether it’s a woman or man or how young or old the victim is, i feel hopeless. it triggers me. it makes me so fucking angry to see people beaten down. it brings me back to my own trauma but it also lights a fucking fire in me, where i get so angry i can’t breathe. does anyone else feel this way? whenever i’m triggered it tends to manifest in anger, but also a really deep sadness. and i feel that right now, i feel really deeply sad for what so many people have to go through and the fact that they aren’t being taken seriously.
                             Label: No
                             Post content: just like the title says i am 58 m. i visit r/depression every day and start by clicking new. while reading the posts i am always looking for those in my age bracket. which is few and far between. my depression has its roots in 2001 but it really took hold during the financial collapse starting in 2008. when i first visited my therapist (about 6 weeks ago) i typed a one page summary of my life from 2001 to current. it wasn't pretty. went from a financially secure, entrepreneur, business owner, great father, decent husband, community volunteer, coach, etc. to...... lost house, broke, in debt, irs issues, depressed mother f-er. the issues were caused by me putting my head in the sand and not being proactive. before, i always took the initiative and created opportunity. slowly but surely i lost that ability and piece by piece my world came crumbling down. so, my story is different than a young person facing depression as a sophomore in college, or a 16 year old afraid to tell their parents about their depression. i care about what they write because i have 2 children (30, 25). and i hope they never experience what we are going through. but, i still read every new post, because i hope to find the answer to my condition. ok, time to go and look at the pile of important things to do, and decide if i am going to take the initiative or go into my typical depressive state. have a good day.
                             Label: Yes
                             Post content: big shoutout to a specific walgreens location for helping me out. long story short: i went to my last prescription of adderall filled at a shitty walgreens location and the head pharmacist treated me like a addict! i went in there with my prescription in hand like normal and the head pharmacist said that they needed to call my doctor (which is understandable) and my doctor told them it was a valid prescription. the pharmacist the tells me they don’t fill prescriptions from out of town (which is funny considering they filled the same prescription a month prior with no issues when she wasn’t there). i tell her that the reason it’s from out of town is because i recently moved here for a new job and no longer lived in the town that was within the same state and 4 hours away. she tells me i need to go to that town if i wanted it filled because no pharmacies in the new town will fill and that the last prescription they filled was a “courtesy” fill. wtf. i then told her that i am dependent on the medication and it isn’t right for her to treat me this way and that i may potential lose my job if i go cold turkey on it. she listens but doesn’t give a shit and i storm out because of her shitty attitude and after she had written stuff that she ended up scribbling out on my prescription. now i am feeling like i am fucked because of her senseless doodling and because my current doctor is retiring in a month and doesn’t have anymore available times for me to come in (and that he is 4 hours away). after i leave, i went to a publix pharmacy and tried to get my prescription filled and the pharmacist there instantly tells me that they are out of stock and that no other publix locations have any in stock. i felt l was getting treated like an addict again considering he read my prescription said adderall and he didn’t want to deal with me. fine. i’ll go to cvs i guess. so i went to cvs and the pharmacist there said that there is a waiting list to become a cvs customer if i want adderall (seriously wtf). so at this point i am freaking out because i recently got a new job and in a month i have a new hire review to access my performance. obviously, going cold turkey on adderall after 4 years would scare me to death when my careers on the line. i figured it wouldn’t hurt to try another walgreens location to try to get it filled. i went in to the new walgreens and and they asked for my id and after that told me it would be done in 15 minutes. no questioning, no treating me like an addict and they even gave me a flu shot while i waited. so glad they helped me out! tl,dr: some pharmacies treat you like addicts and others treat you like a normal human being. always stay close to the ones that treat you with respect.
                             Label: No
                             Post content: so today was not a good day for me. had a meltdown of crying in the kitchen this evening. she groks depression and anxiety, she suffers too, and we help each other out quite a bit whenever we're having a bad day. she walks into the kitchen, examines the boxes of ritz on top of the fridge to make sure she got the butter garlic ones, then walks over to me. the box is turned on its face as she balances it on my head, steadies it, then backs up and asks frankly "well, did it work?" in a very flat and confident tone. my brain goes from "hell no that didn't fix it, you dumb bitch" to "wait, did i miss a memo? is this supposed to fix it somehow?" to "she's either mocking you or trying to make you laugh." laughter won out, as i started laughing and crying simultaneously, the box of crackers falling off my head. i was worried for a split second that they were going to fall, and break, and that would be my fault too, but she caught it. she started saying something like "i couldn't be sure if that would fix it, your manual was translated seven different times in three different languages, this is shoddy work." i don't know why it worked, but given how absurd her approach was, it got my brain out of the bootloop it was stuck in. edit: i don't really think she's ever a bitch, that's just the rapport we have.
                             Label: Yes
                             Post content: hi reddit! my name is amanda morris and i am a professor of chemistry at virginia tech. i serve as an acs expert in the field of sustainable energy this is my second ama on this topic – you can see my biography and that session’s discussion via some of the questions people had last year concerned: how does artificial photosynthesis work; how do the various semiconductor materials used in solar energy work; how does the electrical grid design affect solar energy potentials; how does energy storage technology play a role with solar uptake? i’m happy to answer more about these topics and any others that come to mind. ask me anything about solar energy! i will return at 11 am et (8am pt, 3pm utc) to answer your questions, ask me anything! (10:50 est) hey y'all! i am signed on and replying! i am pumped to see so many comments already! (1:00 est) great time answering your questions! the biggest themes were when is solar cost effective for me, solar cell recycling and life cycle analysis, and tesla's solar relevant technologies. there are calculators to consider the first point - is just one example. first solar is now recycling solar cell modules for large installations - great news! and, tesla continues to push the envelope, but the cost effectiveness of such technoogies isn't proven yet. thanks all! sorry, i didn't get to most of your questions! until next time! -acs edited adding utc and fixing punctuation
                             Label: No

                             Post content: {post}
                             Label:
                             """
            elif disordername == 'post-traumatic stress disorder':  #  disordername == 'post-traumatic stress disorder'
                prompt = f"""Given the content of a post, please determine whether the poster exhibits signs of {disordername}. Respond with either 'Yes' or 'No' only.
                             Here are some examples.
                             Post content: the other night i woke up in the middle of the night to my boyfriend using my hand to jerk himself off. i was horrified. i got up and left, and cried on the couch until i just fell asleep. the next day i confronted him about it. apparently, i had felt him up in my sleep, and he thought i was awake and trying to initiate and he took it too far. he felt horrible. he was on the verge of tears and told me i was completely justified to feel the way i felt and react the way i did, and he offered to sleep on the couch. i believe him, he seemed very genuine. in the years we’ve been together, nothing like this has happened before, and i don’t think it’ll happen again. however... i don’t feel normal around him. it gets significantly harder to breath. i dissociated when talking to him today. my head gets foggy. it gets hard to move. everything feel unreal, and of lower caliber, and numbed, almost like in a dream. i love him, i really do. how do i get back to normal?
                             Label: Yes
                             Post content: i don’t know if this is related to bp, i see a lot of people on here talking about social lives. but i’m just so fucking weird. like awkward, shy, but somehow also saying inappropriate things? being a complete open book this is especially awful in the corporate world.
                             Label: No
                             Post content: hello everyone, do you guys have a recommendation for online therapy services? i have tried betterhelp in the past with two separate counselors. the first counselor left the platform, the second counselor would never give me advice/support/comfort unless it was during a scheduled session. what has worked for you guys in the realm of online therapy?
                             Label: Yes
                             Post content: i was living in a bigger city, and though i knew things were bad, i felt more like i had a chance when i was there. here i just feel hopeless. i am freaking out. and it doesn't help that we live in the middle of nowhere and have no money or support. i have a job interview in a week that i was feeling confident about, and now i feel like i have no chance. and like when i'm here i can't prepare or anything for some reason. not to mention, the rural internet is very very slow so i can't escape to hulu or netflix. i feel like i'm in a prison, but i have no other choice. edit: i have no idea why i put that i was 26. i'm 24. but i guess that's anxiety speaking, feeling like i'm running out of time.
                             Label: No
                             Post content: well it cost me over $15,000, hundreds of therapy hours, a wrong diagnosis, multiple years of effort, and an intense dedication towards becoming a better version of myself..but today i was told by my psychiatrist that it’s time for me to try to manage my life on my own, unaided by mood stabilizers. i’m thrilled, i’m terrified, but most of all i’m proud of myself for fighting so hard to be healthy.
                             Label: Yes
                             Post content: like, playing through what you and someone else would say in the event of some unlikely circumstance? except, it seems likely at the time, that's why you're imagining what's gunna happen, but then it never does, because you don't ever actually talk to anybody, ever? me neither, haha...
                             Label: No
                             Post content: can you have ptsd from living through a traumatic process, and not a single event? for example, sometimes i can't stop thinking about the most awful moments of my mom's last days in hospital. it mostly happens when i am trying to sleep, but sometimes it hits me during the daytime, too, and even though i tell myself to stop thinking about it, i find it difficult to shut these thoughts down and just relax. how can i stop getting these flashbacks? thank you for your time.
                             Label: Yes
                             Post content: does anyone else ever feel this way? i feel like i've been in such a depressing situation for most of my life where i know i have the ability to do well but the lack of focus and the way my thoughts are result in me just never doing well at anything. it feels so isolating. but at least now i have a name for what it all is!
                             Label: No
                             Post content: i am. i cannot articulate most of what triggers me. reactions are anger, fleeing situations, freezing, and crying. just feels like a haze comes over me. the one thing i can't stand, that gets me upset, is being "interrogated" or backed into a corner with questions. not sure why.
                             Label: Yes
                             Post content: hi reddit! my name is amanda morris and i am a professor of chemistry at virginia tech. i serve as an acs expert in the field of sustainable energy this is my second ama on this topic – you can see my biography and that session’s discussion via some of the questions people had last year concerned: how does artificial photosynthesis work; how do the various semiconductor materials used in solar energy work; how does the electrical grid design affect solar energy potentials; how does energy storage technology play a role with solar uptake? i’m happy to answer more about these topics and any others that come to mind. ask me anything about solar energy! i will return at 11 am et (8am pt, 3pm utc) to answer your questions, ask me anything! (10:50 est) hey y'all! i am signed on and replying! i am pumped to see so many comments already! (1:00 est) great time answering your questions! the biggest themes were when is solar cost effective for me, solar cell recycling and life cycle analysis, and tesla's solar relevant technologies. there are calculators to consider the first point - is just one example. first solar is now recycling solar cell modules for large installations - great news! and, tesla continues to push the envelope, but the cost effectiveness of such technoogies isn't proven yet. thanks all! sorry, i didn't get to most of your questions! until next time! -acs edited adding utc and fixing punctuation
                             Label: No
                             
                             Post content: {post}
                             Label:
                             """
            else:
                raise Exception


        # 调用API进行预测
        response = client.chat.completions.create(
            model=model_name,  # 或其他适合的模型
            messages=[
                # 你是一名专业的心理咨询师，了解各类心理疾病症状。你能够根据用户发帖内容判断出用户是否具有某种心理疾病。
                {"role": "system", "content": "You are a professional psychological counselor who understands the symptoms of various psychological disorders. You can determine whether a user has a certain psychological disorder based on their posts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # 输出最可能的结果，保障每次输出的答案稳定性
            top_p=1.0,  # 与temperature=0配合使用
            # max_tokens=1, # 只需要输出：是或者否
            # glm-4模型不支持这两个参数（直接去掉）
            # presence_penalty=0.0,
            # frequency_penalty=0.0
        )

        # 获取预测结果
        result = response.choices[0].message.content.strip()
        # print(f'帖子内容：{post}；\n大模型预测结果为：{result}')
        if disordername == 'suicide ideation':
            if result.lower() in risk_levels:
                return risk_levels[result.lower()]
            else:
                # print(f'错误！大模型返回结果为{result.lower()}')
                for risk_name in risk_levels:
                    if risk_name in result.lower():
                        print(f'根据大模型返回内容，判定风险等级为：{risk_name}')
                        return risk_levels[risk_name]
                return risk_levels['green']  # 对应green，如果是-1会导致没有真实样本。
        else:
            if result.lower() == 'yes':
                return 1
            elif result.lower() == 'no':
                return 0
            else:
                # print(f'错误！大模型返回结果为{result}')
                return 0

    except Exception as e:
        print(f"预测出错：{e}")
        return 0  # 发生错误时返回0


def process_csv(disordername, input_file):
    """
    处理CSV文件，添加预测结果
    Args:
        disordername: 疾病名称
        input_file: 输入CSV文件路径
    """
    try:
        # 读取CSV文件，只选择需要的列
        df = pd.read_csv(input_file, usecols=['post', 'class_id'])

        # 添加预测结果列==>改为有进度条
        # df['pred_class_id'] = df['post'].apply(lambda post: predict_classid(disordername, post))
        tqdm.pandas()  # 启用进度条
        df['pred_class_id'] = df['post'].progress_apply(lambda post: predict_classid(disordername, post))

        gold_labels=df['class_id'].tolist()
        pred_labels=df['pred_class_id'].tolist()

        if disordername == 'suicide ideation':
            acc, p, r, f1 = calculate_f1(gold_labels, pred_labels, average='macro')  # 'weighted'
        else:
            acc, p, r, f1 = calculate_f1(gold_labels, pred_labels, average='macro')  # 'binary'
            # p, r, f1 = calculate_micro_f1(gold_labels, pred_labels)

        print(f'Accuracy: {acc:.5f}\n')
        print(f'P,R,F1:{p:.5f},{r:.5f},{f1:.5f}')

        # 将结果写回原文件
        file_parent = os.path.dirname(input_file)
        file_name = os.path.basename(input_file)

        df.to_csv(file_parent+f'/{model_name}_fewshow/4shot_{model_name}_{file_name}', mode='w', index=False)
        print("预测完成，结果已写入文件")

    except Exception as e:
        print(f"处理CSV文件时出错：{e}")


# 使用示例
if __name__ == "__main__":
    print(f'model_name:{model_name}')
    os.makedirs(f'./data/{model_name}_fewshow', exist_ok=True) # exist_ok=True则文件存在不会报错
    total = len(disorders)
    for i, (disorder, input_file) in enumerate(zip(disorders, input_files), 1):
        print(f"\nProcessing {i}/{total}: {disorder}")
        process_csv(disorder, input_file)

    # process_csv('anxiety', './data/test_anxiety_test.csv')
    # process_csv('suicide ideation', './data/test_self_harm.csv')

