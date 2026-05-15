
CTI_MultiAgent_Project/
│


├── main.py                     # نقطة الانطلاق لتشغيل النظام بالكامل
├── requirements.txt            # المكتبات المطلوبة (LangChain, Streamlit, ChromaDB, etc.)
│

                 => K&A

├── 🎨 ui/                      # 👈 (دور 7: مطور الواجهة)
│   └── app.py                  # واجهة Streamlit أو Gradio التفاعلية
│




├── 🛡️ security/                # 👈 (دور 8: مهندس الحماية - البونص C و D)
│   ├── input_guard.py          # Bonus C: فحص المدخلات لمنع Prompt Injection                  => Omar 
│   └── output_guard.py         # Bonus D: تصفية المخرجات (إخفاء بيانات الشركة الحساسة أو الـ IPs)
│





├── 🧠 core/                    # 👈 (دور 6: مطور الربط - Router)
│   ├── orchestrator.py         # المايسترو الذي يدير المحادثة بين الوكلاء        => Nancy
│   └── router.py               # وكيل التوجيه (يقرر أي Agent سيستلم المهمة)
│




├── 🤖 agents/                  # 👈 (دور 2: مطور الوكلاء)
│   ├── base_agent.py           # الكلاس الأساسي الذي ترث منه باقي الوكلاء
│   ├── osint_agent.py          # وكيل البحث في الإنترنت المفتوح (للبحث عن ثغرات جديدة)      ==> A&K Leader و اخرون 
│   ├── analyst_agent.py        # وكيل التحليل الفني (يفحص IPs, Hashes, MITRE)   
│   └── reporter_agent.py       # وكيل صياغة التقارير النهائية (يلخص النتائج للمستخدم)
│



├── 💾 memory/                  # 👈 (دور 3: مهندس الذاكرة)
│   ├── session_memory.py       # الذاكرة قصيرة المدى (تاريخ المحادثة الحالية)
│   └── vector_memory.py        # الذاكرة طويلة المدى (Vector DB لاسترجاع سياقات قديمة) =>  Mariam A 
│



├── 🛠️ tools/                   # 👈 (دور 4 و 5: مهندسي الـ RAG والأدوات)
│   ├── rag_engine.py           # أداة البحث داخل ملفات الـ PDF التقارير السابقة
│   ├── web_search.py           # أداة البحث الحي في الإنترنت (مثل DuckDuckGo أو Tavily)   ==>  Nesrin
│   └── cag_cache.py            # أداة Caching (تخزين الإجابات السابقة لتسريع النظام)
│


├── 📄 data/                    # 👈 (دور 4: مهندس البيانات)
│   └── raw_reports/            # المجلد الذي تضعون فيه تقارير CTI بصيغة PDF ليقرأها الـ RAG => Mariam  Y 
│




└── ⚙️ config/                  # 👈
    ├── prompts.yaml            # ملف يحتوي على كل الـ Prompts الخاصة بالوكلاء (فصل الكود عن النص) 
    └── settings.yaml           # إعدادات النظام ومفاتيح الـ API   => Abaset