import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pathlib import Path

# CONFIG DASAR 
st.set_page_config(page_title="Visualisasi Data COVID-19", layout="wide")
st.markdown(
    """
    <style>
    .streamlit-expanderHeader[aria-expanded="true"] div:first-child{
        border:2px solid #FF4B4B!important;
        border-radius:6px;
    }
    .streamlit-expanderHeader div:first-child{ padding-left:6px; }
    </style>
    """,
    unsafe_allow_html=True
)

plt.rcParams.update({
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7
})

# LOAD DATA 
@st.cache_data
def load_data():
    fp = Path("covid_19_indonesia_clean.csv")
    if not fp.exists():
        st.stop("❌ File 'covid_19_indonesia_clean.csv' tidak ditemukan.")
    df = pd.read_csv(fp)
    if "Date" not in df.columns:
        st.stop("❌ Kolom 'Date' tidak ada di dataset.")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

st.title("🦠 Visualisasi Data COVID-19 Indonesia per Provinsi")

if "Province" not in df.columns:
    st.stop("❌ Kolom 'Province' tidak ada di dataset.")

# SIDEBAR 
provinsi_list = sorted(df["Province"].dropna().unique())
provinces = st.sidebar.multiselect("Pilih provinsi:", provinsi_list, default=provinsi_list[:3])

st.sidebar.header("📅 Filter Tanggal")
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
start_date = st.sidebar.date_input("Dari tanggal", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("Sampai tanggal", value=max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error("❌ 'Dari tanggal' tidak boleh melebihi 'Sampai tanggal'.")

# FILTER DATA 
mask_tgl = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
mask_prov = df["Province"].isin(provinces)
filtered_df = df[mask_tgl & mask_prov]

# GRAFIK 
show_all = st.checkbox("📑 Tampilkan Grafik Gabungan", value=True)

if show_all:

    with st.expander("📊 Total Kasus per Provinsi", expanded=True):
        bar_data = filtered_df.sort_values("Date").groupby("Province")["Total Cases"].last().reindex(provinces).fillna(0)
        if not bar_data.empty:
            fig, ax = plt.subplots(figsize=(3.5, 2.2))
            bars = ax.bar(bar_data.index, bar_data.values, color="#FFB74D")
            ax.set_ylabel("Total Kasus (orang)")
            ax.set_title("Total Kasus COVID-19 Terbaru")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            plt.xticks(rotation=15)
            for b in bars:
                ax.annotate(f"{int(b.get_height()):,}", (b.get_x()+b.get_width()/2, b.get_height()),
                            textcoords="offset points", xytext=(0,3), ha='center', fontsize=7)
            st.pyplot(fig, clear_figure=True)

    with st.expander("🔵 Scatter Plot Kasus vs Kematian"):
        if {"Total Cases", "Total Deaths"}.issubset(filtered_df.columns):
            scat_df = filtered_df.sort_values("Date").groupby("Province")[["Total Cases", "Total Deaths"]].last().reindex(provinces).dropna()
            if not scat_df.empty:
                fig, ax = plt.subplots(figsize=(3.5, 2.2))
                ax.scatter(scat_df["Total Cases"], scat_df["Total Deaths"], color="#64B5F6")
                for prov, row in scat_df.iterrows():
                    ax.annotate(prov, (row["Total Cases"], row["Total Deaths"]), fontsize=7)
                ax.set_xlabel("Total Kasus")
                ax.set_ylabel("Total Kematian")
                ax.set_title("Total Kasus vs Total Kematian")
                ax.grid(True, linestyle="--", alpha=0.4)
                st.pyplot(fig, clear_figure=True)

    with st.expander("🟣 Pie Chart Total Kasus"):
        pie_df = filtered_df.sort_values("Date").groupby("Province")["Total Cases"].last().reindex(provinces).dropna()
        if not pie_df.empty:
            fig, ax = plt.subplots(figsize=(2.8, 2.8))
            ax.pie(pie_df.values, labels=pie_df.index, autopct="%1.1f%%", startangle=140,
                   colors=sns.color_palette("Pastel1", n_colors=len(pie_df)), shadow=True, textprops={'fontsize': 7})
            ax.set_title("Proporsi Total Kasus", fontsize=9)
            ax.axis("equal")
            st.pyplot(fig, clear_figure=True)

    with st.expander("🟥 Tren Total Kasus Harian (Nasional)"):
        if "Total Cases" in filtered_df.columns:
            area_df = filtered_df.groupby("Date")["Total Cases"].sum().sort_index()
            if not area_df.empty:
                fig, ax = plt.subplots(figsize=(4, 2.3))
                ax.fill_between(area_df.index, area_df.values, color="#E57373", alpha=0.35)
                ax.set_title("Total Kasus Nasional per Hari")
                ax.set_xlabel("Tanggal")
                ax.set_ylabel("Total Kasus")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
                fig.autofmt_xdate(rotation=15, ha='center')
                st.pyplot(fig, clear_figure=True)

    with st.expander("🍃 Distribusi Total Pasien Sembuh per Provinsi"):
        if {"Province", "Total Recovered"}.issubset(filtered_df.columns):
            pie_rec_df = filtered_df.sort_values("Date").groupby("Province")["Total Recovered"].last().reindex(provinces).dropna()
            if not pie_rec_df.empty:
                fig, ax = plt.subplots(figsize=(2.8, 2.8))
                wedges, texts, autotexts = ax.pie(pie_rec_df.values, labels=pie_rec_df.index, autopct='%1.1f%%',
                                                  startangle=140, explode=[0.05]*len(pie_rec_df),
                                                  colors=sns.color_palette("Set3", n_colors=len(pie_rec_df)),
                                                  shadow=True, textprops={'fontsize': 7})
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontweight('bold')
                ax.set_title("Proporsi Total Pasien Sembuh", fontsize=9, color="#388E3C")
                ax.axis("equal")
                st.pyplot(fig, clear_figure=True)

    with st.expander("🔥 Heatmap Korelasi Variabel"):
        cols = [c for c in ["New Cases", "New Deaths", "New Recovered", "Total Cases",
                            "Total Deaths", "Total Recovered", "Total Active Cases"] if c in filtered_df.columns]
        corr_df = filtered_df[cols].dropna()
        if not corr_df.empty and len(cols) >= 2:
            fig, ax = plt.subplots(figsize=(3.5, 2.5))
            sns.heatmap(corr_df.corr(), annot=True, cmap="YlGnBu", linewidths=0.5, ax=ax)
            st.pyplot(fig, clear_figure=True)

# TABEL DATA 
with st.expander("📋 Data Tabel Kasus Harian per Provinsi"):
    if not filtered_df.empty:
        tab_titles = [prov for prov in provinces]
        tabs = st.tabs(tab_titles)
        for tab, prov in zip(tabs, provinces):
            with tab:
                st.subheader(f"Data Kasus Harian – {prov}")
                cols_show = ["Date", "Province", "New Cases", "New Deaths", "New Recovered"]
                show_df = (filtered_df[filtered_df["Province"] == prov][cols_show]
                           .sort_values("Date").reset_index(drop=True))
                st.dataframe(show_df, use_container_width=True)

# FOOTER
st.markdown("📌 *Sumber data:* `covid_19_indonesia_clean.csv`")
