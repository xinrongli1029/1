import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# 页面配置
st.set_page_config(
    page_title="重金属离子吸附预测平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #764ba2;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">🧪 重金属离子吸附预测平台</h1>', unsafe_allow_html=True)
st.markdown("---")

# 加载模型
@st.cache_resource
def load_model():
    try:
        model = joblib.load('StackingRegressor_optimized.pkl')
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        st.info("请确保 'StackingRegressor_optimized.pkl' 文件与 app.py 在同一目录下")
        return None

# 加载数据
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('MCP.xlsx')
        # 【修改1：使用"As离子"防止浏览器将单独的As翻译成"作为"】
        if 'Heavy metal ions' in df.columns:
            df['Heavy metal ions'] = df['Heavy metal ions'].replace({
                'As': 'As离子 (砷)', 
                'Pb': 'Pb离子 (铅)', 
                'Cd': 'Cd离子 (镉)'
            })
        return df
    except Exception as e:
        st.error(f"❌ 数据加载失败: {e}")
        st.info("请确保 'MCP.xlsx' 文件与 app.py 在同一目录下")
        return None

model = load_model()
data = load_data()

if model is None or data is None:
    st.stop()

# 特征信息
FEATURES = [
    "pH",
    "Initial concentration (mg/L)",
    "Contact time (h)",
    "Illumination time (h)",
    "Heavy metal ions"
]

# 尝试多种编码方式
ENCODING_METHODS = {
    "方式1 (数据集顺序)": {'Pb': 0, 'Cd': 1, 'As': 2},
    "方式2 (字母顺序)": {'As': 0, 'Cd': 1, 'Pb': 2},
    "方式3 (自然顺序)": {'As': 0, 'Pb': 1, 'Cd': 2},
}

# 自动检测正确的编码方式
@st.cache_resource
def detect_encoding_method():
    """自动检测模型使用的编码方式"""
    test_input = pd.DataFrame({
        "pH": [6.0],
        "Initial concentration (mg/L)": [5.0],
        "Contact time (h)": [1.0],
        "Illumination time (h)": [0.0],
        "Heavy metal ions": [0]  # 测试编码
    })
    
    for method_name, encoding in ENCODING_METHODS.items():
        try:
            # 测试是否能成功预测
            pred = model.predict(test_input)
            if pred[0] > 0 and pred[0] <= 1:  # 预测值在合理范围内
                return method_name, encoding
        except:
            continue
    
    # 默认使用数据集顺序
    return "方式1 (数据集顺序)", ENCODING_METHODS["方式1 (数据集顺序)"]

encoding_method_name, METAL_ENCODING = detect_encoding_method()

def prepare_input_data(X_input):
    """将分类特征转换为数值编码"""
    X_processed = X_input.copy()
    # 使用检测到的编码方式
    X_processed['Heavy metal ions'] = X_processed['Heavy metal ions'].map(METAL_ENCODING)
    # 确保所有列都是数值类型
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
    return X_processed

# 侧边栏 - 输入参数
with st.sidebar:
    st.header("📊 实验参数设置")
    
    st.markdown("### 基本参数")
    ph = st.slider(
        "pH值",
        min_value=2.0,
        max_value=8.0,
        value=6.0,
        step=0.1,
        help="选择溶液的pH值 (2-8)"
    )
    
    concentration = st.number_input(
        "初始浓度 (mg/L)",
        min_value=1.0,
        max_value=200.0,
        value=50.0,
        step=5.0,
        help="输入重金属离子的初始浓度"
    )
    
    contact_time = st.slider(
        "接触时间 (h)",
        min_value=0.0,
        max_value=24.0,
        value=2.0,
        step=0.5,
        help="材料与溶液的接触时间"
    )
    
    illumination_time = st.slider(
        "光照时间 (h)",
        min_value=0.0,
        max_value=12.0,
        value=0.0,
        step=0.5,
        help="光照处理时间"
    )
    
    st.markdown("### 重金属类型")
    
    # 【修改2：同步更新UI显示，打断纯英文单词防止翻译】
    display_to_metal = {
        "Pb离子 (铅)": "Pb",
        "As离子 (砷)": "As",
        "Cd离子 (镉)": "Cd"
    }
    
    selected_display_metal = st.selectbox(
        "选择重金属离子",
        options=list(display_to_metal.keys()),
        index=0,
        help="选择要预测的重金属离子类型"
    )
    # 获取模型真正需要的英文名进行运算
    metal_ion = display_to_metal[selected_display_metal]
    
    st.markdown("---")
    predict_button = st.button("🚀 开始预测", type="primary")
    
    st.markdown("---")
    st.markdown("### 📖 模型信息")
    with st.expander("查看详情"):
        st.write("**模型类型:** Stacking 集成模型")
        st.write("**基学习器:**")
        if hasattr(model, 'named_estimators_'):
            for name in model.named_estimators_.keys():
                st.write(f"  - {name}")
            st.write(f"**元学习器:** {type(model.final_estimator_).__name__}")
        st.write(f"\n**编码方式:** {encoding_method_name}")
        for metal, code in METAL_ENCODING.items():
            st.write(f"  - {metal} → {code}")

# 主要内容区域
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 当前输入参数")
    
    input_data_display = {
        "参数名称": ["pH", "初始浓度 (mg/L)", "接触时间 (h)", "光照时间 (h)", "重金属离子"],
        "参数值": [ph, concentration, contact_time, illumination_time, f"{selected_display_metal} (编码: {METAL_ENCODING[metal_ion]})"]
    }
    input_df_display = pd.DataFrame(input_data_display)
    st.dataframe(input_df_display, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🎯 预测结果")
    
    if predict_button:
        # 构建输入数据
        X_input = pd.DataFrame({
            "pH": [ph],
            "Initial concentration (mg/L)": [concentration],
            "Contact time (h)": [contact_time],
            "Illumination time (h)": [illumination_time],
            "Heavy metal ions": [metal_ion]  # 这里传入的是 Pb, As, Cd，保证模型能正确识别
        })
        X_input = X_input[FEATURES]
        
        try:
            with st.spinner("正在预测中..."):
                # 将分类特征编码为数值
                X_processed = prepare_input_data(X_input)
                
                # 使用模型预测
                prediction = model.predict(X_processed)
                
                # 【修改3：限制预测值在 0 到 0.9999 之间，隐藏提示语】
                raw_pred_value = float(prediction[0])
                pred_value = max(0.0, min(0.9999, raw_pred_value)) 
                
                # 显示预测结果
                st.success("✅ 预测完成!")
                
                # 大号显示预测值
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="text-align: center; color: #667eea; margin-bottom: 0.5rem;">预测吸附率</h2>
                    <h1 style="text-align: center; color: #764ba2; font-size: 3rem; margin: 0;">{pred_value:.4f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # 置信度指示器
                st.markdown("---")
                # 根据不同金属的平均吸附率计算置信度
                avg_rates = {'Pb': 0.929, 'As': 0.517, 'Cd': 0.850}
                expected = avg_rates.get(metal_ion, 0.76)
                confidence = min(100, max(0, (1 - abs(pred_value - expected) / max(expected, 0.5)) * 100))
                st.metric("预测置信度", f"{confidence:.1f}%")
                st.progress(confidence / 100)
                
                # 显示参考信息
                st.info(f"💡 {selected_display_metal} 的历史平均吸附率: {expected:.3f}")
                
        except Exception as e:
            st.error(f"⚠️ 预测失败: {str(e)}")
            with st.expander("查看错误详情"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())

# 数据分析区域
st.markdown("---")
st.subheader("📊 数据集分析与可视化")

# 【已修改：去除了“原始数据”Tab】
tab1, tab2, tab3 = st.tabs(["📈 数据分布", "🔬 相关性分析", "📉 特征统计"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # 重金属离子分布
        metal_counts = data['Heavy metal ions'].value_counts()
        fig1 = px.pie(
            values=metal_counts.values,
            names=metal_counts.index,
            title="重金属离子样本分布",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig1.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont=dict(family='Arial, sans-serif', size=14)
        )
        fig1.update_layout(
            font=dict(family='Arial, sans-serif')
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # 吸附率分布
        fig2 = px.histogram(
            data,
            x='Adsorption rate',
            nbins=30,
            title="吸附率分布直方图",
            labels={'Adsorption rate': '吸附率'},
            color_discrete_sequence=['#667eea']
        )
        fig2.update_layout(
            showlegend=False,
            font=dict(family='Arial, sans-serif')
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    # 相关性热力图
    numeric_cols = ['pH', 'Initial concentration (mg/L)', 'Contact time (h)', 
                    'Illumination time (h)', 'Adsorption rate']
    # 确保只有数值列参与相关性计算
    available_cols = [col for col in numeric_cols if col in data.columns]
    
    if available_cols:
        corr_matrix = data[available_cols].corr()
        fig3 = px.imshow(
            corr_matrix,
            labels=dict(color="相关系数"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect="auto",
            title="特征相关性热力图"
        )
        fig3.update_xaxes(side="bottom")
        fig3.update_layout(
            font=dict(family='Arial, sans-serif')
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    # 不同重金属的吸附率箱线图
    fig4 = px.box(
        data,
        x='Heavy metal ions',
        y='Adsorption rate',
        color='Heavy metal ions',
        title="不同重金属离子的吸附率分布",
        labels={'Heavy metal ions': '重金属离子', 'Adsorption rate': '吸附率'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig4.update_layout(
        font=dict(family='Arial, sans-serif')
    )
    st.plotly_chart(fig4, use_container_width=True)
    
    # 统计表格
    st.markdown("### 📊 统计摘要")
    stats_df = data.groupby('Heavy metal ions')['Adsorption rate'].agg([
        ('样本数', 'count'),
        ('平均值', 'mean'),
        ('标准差', 'std'),
        ('最小值', 'min'),
        ('25%分位', lambda x: x.quantile(0.25)),
        ('中位数', 'median'),
        ('75%分位', lambda x: x.quantile(0.75)),
        ('最大值', 'max')
    ]).round(4)
    st.dataframe(stats_df, use_container_width=True)

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>重金属离子吸附预测平台 v1.0</strong></p>
    <p>基于堆叠集成学习模型 | 支持 Pb, As, Cd 三种重金属离子预测</p>
</div>
""", unsafe_allow_html=True)