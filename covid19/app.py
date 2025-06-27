import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pathlib import Path

# ---------------- CONFIG DASAR ----------------
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

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    fp = Path("covid_19_indonesia_clean.csv")
    if not fp.exists():
        st.stop("❌ File 'covid_19_indonesia_clean.csv' tidak ditemukan.")
    df = pd.read_csv(fp)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["year"] = df["Date"].dt.year
    else:
        st.warning("Kolom 'Date' tidak ditemukan.")
    return df

df = load_data()

# ---------------- HEADER ----------------
st.title("🦠 Visualisasi Data COVID-19 Indonesia per Provinsi")

if "Province" not in df.columns:
    st.stop("❌ Kolom 'Province' tidak ditemukan di data.")

# ---------------- SIDEBAR ----------------
provinsi_list = sorted(df["Province"].dropna().unique())
provinces = st.sidebar.multiselect(
    "Pilih provinsi yang ingin dianalisis:",
    options=provinsi_list,
    default=provinsi_list[:3] if len(provinsi_list) >= 3 else provinsi_list
)

st.sidebar.header("📅 Filter Tanggal")
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
start_date = st.sidebar.date_input("Dari tanggal", value=min_date,
                                   min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("Sampai tanggal", value=max_date,
                                   min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.sidebar.error("❌ 'Dari tanggal' tidak boleh melebihi 'Sampai tanggal'.")

# ---------------- FILTER DATA ----------------
mask_tgl  = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
mask_prov = df["Province"].isin(provinces)
filtered_df = df[mask_tgl & mask_prov]

# ---------------- GRAFIK ----------------
show_all = st.checkbox("📑 Tampilkan Grafik Gabungan Kasus", value=True)

if show_all:
    # ---------- Bar ----------
    with st.expander("📊 Grafik Per Jenis Kasus", expanded=True):
        bar_data = (
            filtered_df.sort_values("Date")
            .groupby("Province")["Total Cases"]
            .last()
            .reindex(provinces)
            .fillna(0)
        )
        if bar_data.empty:
            st.warning("Data tidak tersedia untuk provinsi terpilih.")
        else:
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(bar_data.index, bar_data.values, color="orange")
            ax.set_ylabel("Total Kasus (orang)")
            ax.set_xlabel("Provinsi")
            ax.set_title("Total Kasus COVID-19 Terbaru")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.xticks(rotation=15)
            for b in bars:
                ax.annotate(f"{int(b.get_height()):,}", xy=(b.get_x()+b.get_width()/2, b.get_height()),
                            xytext=(0,3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
            st.pyplot(fig)

    # ---------- Scatter ----------
    with st.expander("🔵 Scatter Plot Kasus vs Kematian"):
        if {"Total Cases", "Total Deaths"}.issubset(filtered_df.columns):
            scat_df = (
                filtered_df.sort_values("Date")
                .groupby("Province")[["Total Cases", "Total Deaths"]]
                .last()
                .reindex(provinces)
                .dropna()
            )
            if scat_df.empty:
                st.warning("Data tidak tersedia.")
            else:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.scatter(scat_df["Total Cases"], scat_df["Total Deaths"])
                for prov, row in scat_df.iterrows():
                    ax.annotate(prov, (row["Total Cases"], row["Total Deaths"]), fontsize=8)
                ax.set_xlabel("Total Kasus")
                ax.set_ylabel("Total Kematian")
                ax.set_title("Total Kasus vs Total Kematian")
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)
        else:
            st.warning("Kolom yang dibutuhkan tidak lengkap.")

    # ---------- Pie ----------
    with st.expander("🟣 Pie Chart Total Kasus"):
        pie_df = (
            filtered_df.sort_values("Date")
            .groupby("Province")["Total Cases"]
            .last()
            .reindex(provinces)
            .dropna()
        )
        if pie_df.empty:
            st.warning("Data tidak tersedia.")
        else:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(pie_df.values, labels=pie_df.index, autopct="%1.1f%%", startangle=140)
            ax.axis("equal")
            st.pyplot(fig)

    # ---------- Area ----------
    with st.expander("🟩 Area Chart Total Kasus per Hari"):
        if "Total Cases" in filtered_df.columns:
            area_df = (
                filtered_df.groupby("Date")["Total Cases"]
                .sum()
                .sort_index()
            )
            if area_df.empty:
                st.warning("Data tidak tersedia.")
            else:
                fig, ax = plt.subplots(figsize=(4.5, 2.6))
                ax.fill_between(area_df.index, area_df.values, alpha=0.4)
                ax.set_title("Total Kasus Nasional per Hari (Provinsi Terpilih)")
                ax.set_xlabel("Tanggal")
                ax.set_ylabel("Total Kasus")
                ax.grid(True, linestyle="--", alpha=0.5)
                locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
                fig.autofmt_xdate(rotation=15, ha='center')
                st.pyplot(fig)
        else:
            st.warning("Kolom 'Total Cases' tidak ditemukan.")

    # ---------- Heatmap ----------
    with st.expander("🔥 Heatmap Korelasi Kasus"):
        cols = [c for c in ["New Cases", "New Deaths", "Total Cases",
                            "Total Deaths", "Total Active Cases"] if c in filtered_df.columns]
        corr_df = filtered_df[cols].dropna()
        if corr_df.empty or len(cols) < 2:
            st.warning("Data tidak cukup untuk heatmap.")
        else:
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            sns.heatmap(corr_df.corr(), annot=True, cmap="YlGnBu",
                        linewidths=0.5, ax=ax)
            st.pyplot(fig)

# ---------------- TABEL HARIAN (BARU) ----------------
with st.expander("📋 Data Tabel Kasus Harian per Provinsi", expanded=False):
    if filtered_df.empty:
        st.warning("Data tidak tersedia.")
    else:
        # satu tab per provinsi terpilih
        tab_titles = [f"{prov}" for prov in provinces]
        tabs = st.tabs(tab_titles)
        for tab, prov in zip(tabs, provinces):
            with tab:
                st.subheader(f"Data Kasus Harian – {prov}")
                cols_show = ["Date", "Province", "New Cases",
                             "New Deaths", "New Recovered"]
                show_df = (
                    filtered_df[filtered_df["Province"] == prov][cols_show]
                    .sort_values("Date")
                    .reset_index(drop=True)
                )
                st.dataframe(show_df, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("📌 *Sumber data:* `covid_19_indonesia_clean.csv`")
