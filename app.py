import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
from sklearn.cluster import KMeans

# Pillow pour les logos
try:
    from PIL import Image
except ImportError:
    Image = None

# ---------------------------------------------------------
# CONFIG GÉNÉRALE & THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="France Psoriasis – Analyse",
    page_icon="logo_france_psoriasis.png",  # le fichier doit être dans le même dossier que app.py
    layout="wide"
)

# Thème clair + marge haute pour que le header ne soit pas coupé
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
    }
    .block-container {
        padding-top: 2.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Colonnes/Préfixes à exclure des analyses
EXCLUDE_PREFIXES = (
    "alerte", "alertes", "CW", "CW_", "CW_token", "CW_status",
    "CW_firstdate", "CW_firsttime", "CW_finishdate",
    "CW_finishtime"
)
EXCLUDE_COLUMNS = {"nbj", "vague", "fincontact", "mode", "revi"}

# Familles de questions multi-réponses à regrouper par texte
TARGET_MULTI_QIDS = {
    "rs7aa", "rs7ab", "rs7ba", "rs7bb",
    "a3", "b5", "b6", "b10",
    "recrs7aa", "recrs7", "recrs7a"
}

# ---------------------------------------------------------
# FONCTIONS UTILITAIRES
# ---------------------------------------------------------
def get_question_id(name: str) -> str:
    """
    Identifiant 'question' à partir du Name du codebook.
    Exemples :
      - 'b6:1'       -> 'b6'
      - 'rs7ab_1:17' -> 'rs7ab'
    """
    s = str(name)
    if ":" in s:
        s = s.split(":", 1)[0]
    if "_" in s:
        s = s.split("_", 1)[0]
    return s


def app_header():
    """Affiche le bandeau supérieur avec logos et crédits."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if Image is not None:
            try:
                img1 = Image.open("logo_france_psoriasis.png")
                st.image(img1, use_column_width=True)
            except Exception:
                st.markdown("**France Psoriasis**")
        else:
            st.markdown("**France Psoriasis**")

    with col3:
        if Image is not None:
            try:
                img2 = Image.open("logo_cha.png")
                st.image(img2, use_column_width=True)
            except Exception:
                st.markdown("**Centre Hospitalier d’Argenteuil**")
        else:
            st.markdown("**Centre Hospitalier d’Argenteuil**")

    with col2:
        st.markdown(
            """
            <div style="text-align:center;">
              <div style="font-size:22px; font-weight:bold; margin-bottom:4px;">
                Psoriasis
              </div>
              <div style="font-size:17px; margin-bottom:4px;">
                Centre Hospitalier d’Argenteuil & Association France Psoriasis
              </div>
              <div style="font-size:14px; color:#555;">
                Site conçu par <b>Dr Dorra MEDHAFFAR</b> & <b>Pr Emmanuel Mahé</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")


@st.cache_data
def load_data():
    """Charge le fichier Excel BDD_pso.xlsx et prépare Codebook + Data."""
    xls = pd.ExcelFile("BDD_pso.xlsx")
    sheetnames = {s.lower(): s for s in xls.sheet_names}

    if "codebook" not in sheetnames or "data" not in sheetnames:
        raise ValueError("Les onglets 'Codebook' et 'Data' sont requis dans le fichier Excel.")

    cb = pd.read_excel(xls, sheetnames["codebook"])
    df = pd.read_excel(xls, sheetnames["data"])

    cb = cb.dropna(subset=["Name"])
    cb["Name"] = cb["Name"].astype(str)
    cb["Description"] = cb["Description"].astype(str)
    cb["Value"] = cb["Value"].astype(str)

    # Exclure colonnes techniques dans Data
    cols_keep = []
    for c in df.columns:
        c_str = str(c)
        if c_str in EXCLUDE_COLUMNS:
            continue
        if any(c_str.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        cols_keep.append(c)
    df = df[cols_keep].copy()

    # Dictionnaires Codebook
    name_to_desc = dict(zip(cb["Name"], cb["Description"]))
    name_to_value = {
        n: (v if v.lower() != "nan" and v != "" else None)
        for n, v in zip(cb["Name"], cb["Value"])
    }

    cb["question_id"] = cb["Name"].apply(get_question_id)

    # question_id -> colonnes présentes dans Data
    question_to_cols = {}
    for name in cb["Name"]:
        if name in df.columns:
            qid = get_question_id(name)
            if any(qid.startswith(p) for p in EXCLUDE_PREFIXES):
                continue
            question_to_cols.setdefault(qid, []).append(name)

    for k in question_to_cols:
        question_to_cols[k] = sorted(question_to_cols[k])

    return cb, df, name_to_desc, name_to_value, question_to_cols


try:
    codebook, data, name_to_desc, name_to_value, question_to_cols = load_data()
except FileNotFoundError:
    st.error("❌ Fichier 'BDD_pso.xlsx' introuvable. Place-le dans le même dossier que 'app.py'.")
    st.stop()
except Exception as e:
    st.error(f"❌ Erreur de chargement : {e}")
    st.stop()


def label_with_desc(name: str) -> str:
    """Affiche 'Name – Description' s'il existe dans le Codebook."""
    desc = name_to_desc.get(str(name))
    if desc:
        return f"{name} – {desc}"
    return str(name)


def clean_categorical_series(s: pd.Series) -> pd.Series:
    """
    Nettoie une série catégorielle :
    - '0' -> 'Non'
    - '1' -> 'Oui'
    - NaN / vide -> 'Non'
    """
    s = s.astype(str).str.strip()
    return s.replace(
        {"0": "Non", "1": "Oui", "nan": "Non", "": "Non"}
    ).fillna("Non")


def get_numeric_candidates(df: pd.DataFrame):
    """Détecte les variables numériques pertinentes."""
    numeric_cols = []
    for c in df.columns:
        if c in EXCLUDE_COLUMNS or any(str(c).startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().sum() > len(df) * 0.4 and col.nunique() > 5:
            numeric_cols.append(c)
    return numeric_cols


def is_binary_column(df: pd.DataFrame, col: str) -> bool:
    """Retourne True si la colonne est essentiellement binaire (0/1)."""
    s = df[col].dropna().astype(str).str.strip()
    if s.empty:
        return False
    uniq = set(s.unique())
    return uniq.issubset({"0", "1"})


def compute_question_label(cols):
    """Construit un libellé commun pour une question multi-réponses."""
    descs = [name_to_desc.get(c, c) for c in cols]
    if not descs:
        return cols[0]
    base = descs[0]
    for d in descs[1:]:
        i = 0
        while i < len(base) and i < len(d) and base[i] == d[i]:
            i += 1
        base = base[:i]
    qpos = base.rfind("?")
    if qpos != -1:
        base = base[: qpos + 1]
    base = base.strip(" :-(")
    if len(base) < 10:
        return descs[0]
    return base


def build_comorbidity_count(df: pd.DataFrame, question_id: str):
    """
    Calcule un score de comorbidités pour un groupe de colonnes
    (question multi-éléments binaire).
    """
    cols = question_to_cols.get(question_id, [])
    if not cols:
        return None
    sub = df[cols].astype(str)
    present = sub.apply(lambda col: col.notna() & (col.str.strip() != "") & (col.str.strip() != "0"))
    return present.sum(axis=1)


def build_multi_question_groups():
    """
    Construit la liste des groupes multi-réponses :
    - pour les qids dans TARGET_MULTI_QIDS : regroupement par texte de question (avant '?')
    - pour les autres qids : multi si plusieurs colonnes binaires 0/1

    Retourne : (groups, used_cols)
        groups : liste de dict {label, qid, cols}
        used_cols : ensemble de toutes les colonnes appartenant à un groupe
    """
    groups = []
    used_cols = set()

    # 1. Groupes forcés (liste donnée par l'utilisatrice)
    for qid, cols in question_to_cols.items():
        if qid not in TARGET_MULTI_QIDS:
            continue
        base_map = {}
        for col in cols:
            desc = name_to_desc.get(col, str(col))
            if "?" in desc:
                base = desc.split("?", 1)[0].strip() + " ?"
            else:
                base = desc.strip()
            base_map.setdefault(base, []).append(col)

        for base, cols_group in base_map.items():
            groups.append(
                {
                    "label": base,
                    "qid": qid,
                    "cols": sorted(cols_group),
                }
            )
            used_cols.update(cols_group)

    # 2. Autres groupes vraiment binaires (0/1)
    for qid, cols in question_to_cols.items():
        if qid in TARGET_MULTI_QIDS:
            continue
        if len(cols) > 1 and all(is_binary_column(data, c) for c in cols):
            label = compute_question_label(cols)
            groups.append(
                {
                    "label": label,
                    "qid": qid,
                    "cols": sorted(cols),
                }
            )
            used_cols.update(cols)

    return groups, used_cols


def build_question_options_for_descriptive():
    """
    Construit les options pour le selectbox descriptif :
    - questions multi-réponses (une par question) -> kind = "multi"
    - autres variables -> kind = "single"
    """
    multi_groups, multi_cols = build_multi_question_groups()
    options = []

    for g in multi_groups:
        options.append(
            {
                "label": g["label"],
                "kind": "multi",
                "id": g["qid"],
                "cols": g["cols"],
            }
        )

    # Variables simples (non utilisées dans un groupe multi)
    for col in data.columns:
        if col in multi_cols:
            continue
        desc = name_to_desc.get(col)
        if not desc:
            continue
        options.append(
            {
                "label": desc,
                "kind": "single",
                "id": col,
                "cols": [col],
            }
        )

    options = sorted(options, key=lambda x: x["label"])
    return options


def cramers_v_from_table(ct: pd.DataFrame) -> float:
    """Calcule le V de Cramér à partir d'une table de contingence."""
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    if n == 0:
        return np.nan
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))


# ---------------------------------------------------------
# HEADER & NAVIGATION
# ---------------------------------------------------------
app_header()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Accueil / Résumé global",
        "📊 Analyse descriptive",
        "🧪 Analyse analytique",
        "🧬 Exploration avancée (clustering)",
        "📖 Comprendre le psoriasis",
        "📝 Discussion scientifique",
        "📚 Hypothèses & pré-traitement",
    ],
)

# ---------------------------------------------------------
# 1. ACCUEIL / RÉSUMÉ GLOBAL
# ---------------------------------------------------------
if page == "🏠 Accueil / Résumé global":
    st.title("Résumé global de l’enquête France Psoriasis")

    n = len(data)
    st.subheader("Effectif de l’étude")
    st.markdown(f"- Nombre total de répondants : **{n}**")

    # Sexe
    if "s1" in data.columns:
        st.subheader(label_with_desc("s1"))
        sex_counts = data["s1"].value_counts().rename_axis("Sexe").reset_index(name="Effectif")
        st.dataframe(sex_counts)
        fig_sex = px.pie(sex_counts, values="Effectif", names="Sexe", title="Répartition par sexe")
        st.plotly_chart(fig_sex, use_container_width=True)

    # Âge
    if "xs2" in data.columns:
        st.subheader(label_with_desc("xs2"))
        ages = pd.to_numeric(data["xs2"], errors="coerce")
        st.markdown(
            f"- Âge moyen : **{ages.mean():.1f} ans**  \n"
            f"- Médiane : **{ages.median():.1f} ans**  \n"
            f"- Intervalle : **{ages.min():.0f} – {ages.max():.0f} ans**"
        )
        fig_age = px.histogram(ages, nbins=20, title="Distribution des âges")
        st.plotly_chart(fig_age, use_container_width=True)

    # Habitat
    if "s7c" in data.columns:
        st.subheader(label_with_desc("s7c"))
        habitat = data["s7c"].value_counts().reset_index()
        habitat.columns = ["Habitat", "Effectif"]
        st.dataframe(habitat)
        fig_hab = px.bar(
            habitat,
            x="Habitat",
            y="Effectif",
            title="Type d’habitat",
            labels={"Habitat": "", "Effectif": "Effectif"},
        )
        fig_hab.update_layout(xaxis_tickangle=30)
        st.plotly_chart(fig_hab, use_container_width=True)

    # Situation pro / familiale
    for col in ["rs4", "rs5"]:
        if col in data.columns:
            st.subheader(label_with_desc(col))
            ser = data[col].dropna()
            counts = ser.value_counts().reset_index()
            counts.columns = ["Modalité", "Effectif"]
            st.dataframe(counts)
            fig = px.bar(counts, x="Modalité", y="Effectif", title=label_with_desc(col))
            fig.update_layout(xaxis_tickangle=30)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        🔍 Les autres pages permettent d'explorer :
        - **Analyse descriptive** : distributions, regroupement des questions, carte par région  
        - **Analyse analytique** : profils, chi², corrélations catégorielles  
        - **Exploration avancée** : clustering des profils de patients  
        - **Comprendre le psoriasis** : rappel clinique & données françaises  
        - **Discussion scientifique** : interprétation détaillée des résultats  
        - **Hypothèses & pré-traitement** : transparence méthodologique  
        """
    )

# ---------------------------------------------------------
# 2. ANALYSE DESCRIPTIVE
# ---------------------------------------------------------
elif page == "📊 Analyse descriptive":
    st.title("Analyse descriptive détaillée")

    tab_var, tab_map = st.tabs(["Exploration des questions", "Carte géographique"])

    # ------------------ EXPLORATION DES QUESTIONS ------------------
    with tab_var:
        st.subheader("Exploration par question / item")

        question_options = build_question_options_for_descriptive()
        if not question_options:
            st.warning("Aucune question exploitable détectée.")
        else:
            labels = [o["label"] for o in question_options]
            selected_label = st.selectbox(
                "Choisissez une question ou un item à explorer",
                options=labels,
            )
            choice = next(o for o in question_options if o["label"] == selected_label)

            kind = choice["kind"]
            cols = choice["cols"]

            st.markdown(f"**Question / item sélectionné :** {selected_label}")
            st.markdown(f"**Nombre d’items / réponses possibles :** {len(cols)}")

            graph_type = st.radio(
                "Type de graphique",
                ["Barres", "Camembert", "Histogramme (si question numérique)"],
                horizontal=True,
            )

            # ---- CAS SINGLE ----
            if kind == "single":
                var = cols[0]
                serie = data[var]
                numeric_try = pd.to_numeric(serie, errors="coerce")
                is_numeric = (
                        numeric_try.notna().sum() > len(data) * 0.5
                        and numeric_try.nunique() > 5
                )

                if is_numeric:
                    st.markdown("#### Variable numérique")
                    st.write(numeric_try.describe())
                    if graph_type == "Histogramme (si question numérique)":
                        fig = px.histogram(numeric_try, nbins=20, title=selected_label)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.box(
                            numeric_try,
                            points="outliers",
                            title=selected_label,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### Variable catégorielle")
                    serie_clean = clean_categorical_series(serie)
                    counts = serie_clean.value_counts().reset_index()
                    counts.columns = ["Modalité", "Effectif"]
                    st.dataframe(counts)

                    if graph_type == "Barres":
                        fig = px.bar(
                            counts,
                            x="Modalité",
                            y="Effectif",
                            title=selected_label,
                        )
                        fig.update_layout(xaxis_tickangle=30)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.pie(
                            counts,
                            values="Effectif",
                            names="Modalité",
                            title=selected_label,
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # ---- CAS MULTI-RÉPONSES ----
            else:
                st.markdown("#### Question à réponses multiples")

                rows = []
                for col in cols:
                    col_ser = data[col].astype(str).str.strip()
                    present = col_ser.notna() & (col_ser != "") & (col_ser != "0")
                    count = present.sum()
                    if count == 0:
                        continue
                    label = name_to_value.get(col) or name_to_desc.get(col) or col
                    rows.append((label, count))

                if not rows:
                    st.info("Aucune réponse positive pour cette question.")
                else:
                    summary = pd.DataFrame(rows, columns=["Réponse", "Effectif"]).sort_values(
                        "Effectif", ascending=False
                    )
                    summary["%"] = summary["Effectif"] / len(data) * 100
                    st.dataframe(summary)

                    if graph_type == "Barres":
                        fig = px.bar(
                            summary,
                            x="Réponse",
                            y="Effectif",
                            title=selected_label,
                            labels={"Réponse": "", "Effectif": "Effectif"},
                        )
                        fig.update_layout(xaxis_tickangle=30)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.pie(
                            summary,
                            values="Effectif",
                            names="Réponse",
                            title=selected_label,
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # ------------------ CARTE GÉOGRAPHIQUE ------------------
    with tab_map:
        st.subheader("Carte géographique – répartition par grandes régions")

        region_coords = {
            "Ile de France": (48.85, 2.35),
            "Île de France": (48.85, 2.35),
            "Ouest (Pays de la Loire, Bretagne, Poitou-Charentes)": (47.5, -1.5),
            "Méditerranée (Languedoc-Roussillon, PACA, Corse)": (43.5, 4.8),
            "Mediterranee (Languedoc-Roussillon, PACA, Corse)": (43.5, 4.8),
            "Bassin parisien Ouest (Haute-Normandie, Basse Normandie, Centre)": (48.6, 0.2),
            "Bassin parisien Est (Champagne-Ardenne, Picardie, Bourgogne)": (48.7, 3.5),
            "Sud-Ouest (Aquitaine, Midi-Pyrénées, Limousin)": (44.5, 0.2),
            "Est (Lorraine, Alsace, Franche-Comté)": (48.5, 6.5),
            "Nord (Nord-Pas-de-Calais)": (50.5, 2.7),
            "Sud-Est (Rhône-Alpes, Auvergne)": (45.5, 4.5),
        }

        if "qs3c" in data.columns:
            region_ser = data["qs3c"].dropna().astype(str).str.strip()
            counts = region_ser.value_counts()
            if counts.empty:
                st.warning("Pas de régions exploitables dans qs3c.")
            else:
                rows = []
                for region, eff in counts.items():
                    coord = region_coords.get(region)
                    if coord is None:
                        reg_norm = (
                            region.replace("é", "e").replace("è", "e").replace("ê", "e")
                        )
                        coord = region_coords.get(reg_norm)
                    if coord is None:
                        continue
                    lat, lon = coord
                    rows.append(
                        {"Région": region, "Effectif": eff, "lat": lat, "lon": lon}
                    )

                st.markdown("#### Effectifs par grande région (qs3c)")
                st.dataframe(
                    counts.rename("Effectif").reset_index().rename(
                        columns={"index": "Région"}
                    )
                )

                if not rows:
                    st.warning(
                        "Impossible d'associer des coordonnées aux régions (vérifier les libellés)."
                    )
                else:
                    df_map = pd.DataFrame(rows)
                    fig_map = px.scatter_mapbox(
                        df_map,
                        lat="lat",
                        lon="lon",
                        size="Effectif",
                        hover_name="Région",
                        hover_data={"Effectif": True},
                        zoom=4.5,
                        height=600,
                    )
                    fig_map.update_layout(
                        mapbox_style="open-street-map",
                        mapbox_center={"lat": 46.5, "lon": 2.5},
                        title="Répartition des répondants par grandes régions (qs3c)",
                    )
                    st.plotly_chart(fig_map, use_container_width=True)

        elif "qs3" in data.columns:
            st.info(
                "Aucune variable de région détaillée (qs3c) trouvée. Affichage par IDF / Province via qs3."
            )
            ser = data["qs3"].dropna().astype(str).str.strip()
            counts = ser.value_counts().reset_index()
            counts.columns = ["Zone", "Effectif"]
            st.dataframe(counts)
            fig_bar = px.bar(
                counts,
                x="Zone",
                y="Effectif",
                title="Répartition IDF / Province (qs3)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning(
                "Aucune variable géographique exploitable (qs3c ou qs3) n'a été trouvée."
            )

# ---------------------------------------------------------
# 3. ANALYSE ANALYTIQUE
# ---------------------------------------------------------
elif page == "🧪 Analyse analytique":
    st.title("Analyse analytique")

    tab_profils, tab_tests, tab_global = st.tabs(
        ["Profils par question (Oui/Non)", "Tests personnalisés", "Analyses globales (catégorielles)"]
    )

    numeric_cols = get_numeric_candidates(data)
    cat_cols = [c for c in data.columns if c not in numeric_cols]

    # ---------------- PROFILS PAR QUESTION MULTI ----------------
    with tab_profils:
        st.subheader("Profils selon une question multi-réponses (Oui / Non)")

        if not numeric_cols:
            st.info("Aucune variable numérique pertinente détectée.")
        else:
            default_num = "xs2" if "xs2" in numeric_cols else numeric_cols[0]
            num_var = st.selectbox(
                "Variable numérique de profil (ex : âge)",
                options=numeric_cols,
                index=numeric_cols.index(default_num),
                format_func=label_with_desc,
            )

            multi_groups, _ = build_multi_question_groups()
            multi_groups = sorted(multi_groups, key=lambda x: x["label"])

            if not multi_groups:
                st.warning(
                    "Aucune question multi-réponses identifiée pour cette analyse."
                )
            else:
                labels = [g["label"] for g in multi_groups]
                selected_label = st.selectbox(
                    "Question à analyser", options=labels
                )
                group = next(g for g in multi_groups if g["label"] == selected_label)

                group_cols = group["cols"]
                st.markdown(f"**Question sélectionnée :** {selected_label}")
                st.markdown(f"**Nombre d’items :** {len(group_cols)}")

                results = []
                num_series = pd.to_numeric(data[num_var], errors="coerce")

                for col in group_cols:
                    col_raw = data[col].astype(str).str.strip()
                    yes_mask = col_raw.notna() & (col_raw != "") & (col_raw != "0")
                    no_mask = ~yes_mask

                    n_yes = yes_mask.sum()
                    n_no = no_mask.sum()

                    if n_yes < 5 or n_no < 5:
                        continue

                    num_yes = num_series[yes_mask]
                    num_no = num_series[no_mask]

                    mean_yes = num_yes.mean()
                    mean_no = num_no.mean()

                    t_stat, p_val = stats.ttest_ind(
                        num_yes.dropna(),
                        num_no.dropna(),
                        equal_var=False,
                        nan_policy="omit",
                    )

                    label = name_to_value.get(col) or name_to_desc.get(col) or col
                    results.append(
                        {
                            "Item": label,
                            "N Oui": n_yes,
                            "N Non": n_no,
                            f"Moyenne {label_with_desc(num_var)} – Oui": mean_yes,
                            f"Moyenne {label_with_desc(num_var)} – Non": mean_no,
                            "p-value (t-test)": p_val,
                        }
                    )

                if not results:
                    st.info(
                        "Pas assez de données pour calculer des profils (effectifs trop faibles)."
                    )
                else:
                    df_prof = pd.DataFrame(results).sort_values("p-value (t-test)")
                    st.markdown("#### Résultats (classés par p-value croissante)")
                    st.dataframe(df_prof)
                    st.markdown(
                        """
                        🧾 **Interprétation**  
                        - Une p-value < 0,05 suggère une différence significative de la variable choisie
                          (par ex. l'âge) entre les groupes **Oui** et **Non** pour l’item considéré.  
                        - Cela permet d’identifier, par exemple, les comorbidités plus fréquentes chez les patients plus âgés.
                        """
                    )

    # ---------------- TESTS PERSONNALISÉS ----------------
    with tab_tests:
        st.subheader("Tests statistiques personnalisés")

        test_type = st.radio(
            "Choisir un test",
            ["Chi² (2 variables catégorielles)", "Comparaison de moyennes (numérique vs 2 groupes)"],
        )

        if test_type == "Chi² (2 variables catégorielles)":
            if len(cat_cols) < 2:
                st.warning("Pas assez de variables catégorielles.")
            else:
                v1 = st.selectbox("Variable 1", cat_cols, format_func=label_with_desc)
                v2_candidates = [c for c in cat_cols if c != v1]
                v2 = st.selectbox("Variable 2", v2_candidates, format_func=label_with_desc)

                ser1 = clean_categorical_series(data[v1])
                ser2 = clean_categorical_series(data[v2])
                ct = pd.crosstab(ser1, ser2)

                st.markdown("#### Table de contingence")
                st.dataframe(ct)

                chi2, p, dof, expected = stats.chi2_contingency(ct)
                st.markdown("#### Résultats du test du Chi²")
                st.write(f"- Chi² = **{chi2:.3f}**")
                st.write(f"- ddl = **{dof}**")
                st.write(f"- p-value = **{p:.4f}**")

                if p < 0.05:
                    st.success("Association statistiquement significative (p < 0,05).")
                else:
                    st.info("Pas d'association significative mise en évidence.")

        else:  # Comparaison de moyennes
            if not numeric_cols or not cat_cols:
                st.warning(
                    "Il faut au moins 1 variable numérique et 1 variable catégorielle."
                )
            else:
                num_var = st.selectbox(
                    "Variable numérique", numeric_cols, format_func=label_with_desc
                )
                cat_var = st.selectbox(
                    "Variable de groupe (2 modalités)",
                    cat_cols,
                    format_func=label_with_desc,
                )

                ser_num = pd.to_numeric(data[num_var], errors="coerce")
                ser_cat = clean_categorical_series(data[cat_var])
                df_sub = pd.DataFrame({num_var: ser_num, cat_var: ser_cat}).dropna()

                mods = df_sub[cat_var].unique()
                if len(mods) < 2:
                    st.warning("La variable de groupe doit avoir au moins 2 modalités.")
                else:
                    m1 = st.selectbox("Modalité 1", mods)
                    m2 = st.selectbox("Modalité 2", [m for m in mods if m != m1])

                    g1 = df_sub[df_sub[cat_var] == m1][num_var]
                    g2 = df_sub[df_sub[cat_var] == m2][num_var]

                    st.markdown("#### Statistiques descriptives")
                    stats_df = pd.DataFrame(
                        {
                            "Groupe": [m1, m2],
                            "N": [g1.count(), g2.count()],
                            "Moyenne": [g1.mean(), g2.mean()],
                            "Écart-type": [g1.std(), g2.std()],
                        }
                    )
                    st.dataframe(stats_df)

                    t, p_val = stats.ttest_ind(
                        g1, g2, equal_var=False, nan_policy="omit"
                    )
                    st.markdown("#### Résultats du test t de Student (Welch)")
                    st.write(f"- t = **{t:.3f}**")
                    st.write(f"- p-value = **{p_val:.4f}**")

                    if p_val < 0.05:
                        st.success("Différence statistiquement significative (p < 0,05).")
                    else:
                        st.info("Pas de différence significative détectée.")

    # ---------------- ANALYSES GLOBALES CATEGORIELLES ----------------
    with tab_global:
        st.subheader("Analyses globales sur les variables catégorielles")

        # Construction de variables synthétiques intéressantes
        cat_vars = {}

        if "s1" in data.columns:
            cat_vars["Sexe"] = clean_categorical_series(data["s1"])
        if "rs4" in data.columns:
            cat_vars["Situation professionnelle"] = clean_categorical_series(data["rs4"])
        if "rs5" in data.columns:
            cat_vars["Situation familiale"] = clean_categorical_series(data["rs5"])
        if "qs3c" in data.columns:
            cat_vars["Région"] = clean_categorical_series(data["qs3c"])

        # Score de comorbidités rs7aa (ou rs7ab)
        comor = build_comorbidity_count(data, "rs7aa")
        if comor is None:
            comor = build_comorbidity_count(data, "rs7ab")
        if comor is not None:
            cat_vars["≥3 comorbidités"] = pd.Series(
                np.where(comor >= 3, "≥3 comorbidités", "<3 comorbidités")
            )

        if len(cat_vars) < 2:
            st.info("Pas assez de variables catégorielles pour une analyse globale.")
        else:
            labels = list(cat_vars.keys())

            # Matrice de V de Cramér
            m = pd.DataFrame(index=labels, columns=labels, dtype=float)
            for i, l1 in enumerate(labels):
                for j, l2 in enumerate(labels):
                    if i == j:
                        m.loc[l1, l2] = 1.0
                    elif i > j:
                        m.loc[l1, l2] = m.loc[l2, l1]
                    else:
                        s1 = cat_vars[l1]
                        s2 = cat_vars[l2]
                        df_ = pd.DataFrame({l1: s1, l2: s2}).dropna()
                        if df_.empty:
                            v = np.nan
                        else:
                            ct = pd.crosstab(df_[l1], df_[l2])
                            if ct.shape[0] < 2 or ct.shape[1] < 2:
                                v = np.nan
                            else:
                                v = cramers_v_from_table(ct)
                        m.loc[l1, l2] = v

            st.markdown("#### Corrélations catégorielles (V de Cramér)")
            st.dataframe(m)

            fig_heat = px.imshow(
                m.astype(float),
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Reds",
                title="Matrice de V de Cramér (variables catégorielles clés)",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown(
                """
                🔎 **Lecture rapide**  
                - V de Cramér proche de 0 → faible association.  
                - V de Cramér > 0,2 → association modérée potentiellement intéressante.  

                Cela permet par exemple de voir si la situation professionnelle ou familiale
                est liée au fait d’avoir ≥3 comorbidités ou à la répartition géographique.
                """
            )

            # Quelques tests Chi² automatiques utiles pour la discussion
            st.markdown("#### Quelques tests du Chi² pré-sélectionnés")

            tests_pairs = [
                ("Sexe", "≥3 comorbidités"),
                ("Situation professionnelle", "≥3 comorbidités"),
                ("Situation familiale", "≥3 comorbidités"),
                ("Région", "≥3 comorbidités"),
            ]

            for v1, v2 in tests_pairs:
                if v1 not in cat_vars or v2 not in cat_vars:
                    continue
                st.markdown(f"**{v1} × {v2}**")
                s1 = cat_vars[v1]
                s2 = cat_vars[v2]
                df_ = pd.DataFrame({v1: s1, v2: s2}).dropna()
                if df_.empty:
                    st.write("Données insuffisantes.")
                    continue
                ct = pd.crosstab(df_[v1], df_[v2])
                st.dataframe(ct)
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    st.write("Pas assez de modalités pour un Chi² interprétable.")
                    st.markdown("---")
                    continue
                chi2, p, dof, _ = stats.chi2_contingency(ct)
                st.write(f"- Chi² = **{chi2:.2f}**, ddl = **{dof}**, p-value = **{p:.4f}**")
                st.markdown("---")

# ---------------------------------------------------------
# 4. CLUSTERING
# ---------------------------------------------------------
elif page == "🧬 Exploration avancée (clustering)":
    st.title("Exploration avancée – Clustering des profils de patients")

    st.markdown(
        """
        L’objectif est d’identifier des **profils de patients** à partir de deux dimensions :
        - l’**âge** (xs2)  
        - le **nombre de comorbidités chroniques** déclarées (question des maladies associées)

        On applique un algorithme de **K-means** pour regrouper les patients en clusters.
        """
    )

    question_for_comor = "rs7aa" if "rs7aa" in question_to_cols else "rs7ab"
    comorbid_count = build_comorbidity_count(data, question_for_comor)

    if comorbid_count is None:
        st.warning(
            "Impossible de calculer le score de comorbidités (groupe rs7aa/rs7ab absent)."
        )
    elif "xs2" not in data.columns:
        st.warning("La variable d'âge 'xs2' est requise pour le clustering.")
    else:
        ages = pd.to_numeric(data["xs2"], errors="coerce")
        df_cluster = pd.DataFrame({"age": ages, "nb_comorbidites": comorbid_count}).dropna()

        st.markdown(f"- Nombre de patients utilisables pour le clustering : **{len(df_cluster)}**")

        if len(df_cluster) < 20:
            st.warning("Échantillon trop petit pour un clustering informatif (> 20 recommandé).")
        else:
            k = st.slider("Nombre de clusters (K)", min_value=2, max_value=5, value=3)

            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(df_cluster)
            df_cluster["cluster"] = labels.astype(str)

            fig = px.scatter(
                df_cluster,
                x="age",
                y="nb_comorbidites",
                color="cluster",
                title="Clusters de patients (âge vs nombre de comorbidités)",
                labels={"age": "Âge", "nb_comorbidites": "Nombre de comorbidités chroniques"},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Résumé statistique par cluster")
            summary = (
                df_cluster.groupby("cluster")
                .agg(
                    N=("age", "count"),
                    age_moy=("age", "mean"),
                    comor_moy=("nb_comorbidites", "mean"),
                )
                .reset_index()
            )
            st.dataframe(summary)

            st.markdown(
                """
                🔍 **Interprétation possible des clusters :**
                - Un cluster peut regrouper des patients **plus jeunes avec peu ou pas de comorbidités**.  
                - Un autre cluster peut correspondre à des patients **plus âgés, très comorbides**, 
                  potentiellement à plus haut risque cardiovasculaire ou métabolique.  
                - Un cluster intermédiaire peut représenter des profils mixtes.  

                Ces profils peuvent aider à :
                - cibler des messages de **prévention** (poids, tabac, diabète, activité physique),  
                - prioriser certains patients pour des **évaluations spécialisées** (cardio, rhumato, psy),  
                - soutenir des arguments de **santé publique** sur la nature systémique du psoriasis.
                """
            )

# ---------------------------------------------------------
# 5. COMPRENDRE LE PSORIASIS
# ---------------------------------------------------------
elif page == "📖 Comprendre le psoriasis":
    st.title("Comprendre le psoriasis")

    st.markdown(
        """
        Le **psoriasis** est une maladie inflammatoire chronique de la peau, à médiation immune, 
        qui touche la peau, parfois les ongles et les articulations, et s’accompagne fréquemment 
        de comorbidités métaboliques et cardiovasculaires.

        ---
        ### 1. Physiopathologie (en bref)

        - Activation de l’axe **IL-23 / IL-17 / TNFα** et des lymphocytes T.  
        - Hyperprolifération des kératinocytes → plaques érythémato-squameuses bien limitées.  
        - Inflammation systémique de bas grade impliquée dans le risque cardio-métabolique.  

        ---
        ### 2. Formes cliniques principales

        - **Psoriasis en plaques** (forme la plus fréquente).  
        - Psoriasis du **cuir chevelu**.  
        - Atteinte **unguéale** (ongles striés, dépressés, onycholyse).  
        - Psoriasis **palmo-plantaire**.  
        - Psoriasis **en gouttes**.  
        - Formes sévères : **pustuleux généralisé**, **érythrodermie psoriasique**.  
        - **Rhumatisme psoriasique** (atteinte articulaire périphérique ou axiale).

        ---
        ### 3. Comorbidités associées

        Le psoriasis est reconnu comme une **maladie systémique** :

        - **Syndrome métabolique** : obésité, diabète de type 2, dyslipidémies.  
        - **Hypertension artérielle**, maladie coronarienne, AVC.  
        - **Rhumatisme psoriasique** (douleurs, raideurs, dactylites, enthésites).  
        - **Troubles anxieux et dépressifs**, altération de l’estime de soi.  
        - Autres associations possibles : NAFLD, MICI, uvéites.

        ---
        ### 4. Impact sur la qualité de vie

        - Gêne esthétique, stigmatisation, sentiment de rejet.  
        - Répercussions sur la vie professionnelle, sociale, affective et intime.  
        - Peur des poussées, incompréhension de l’entourage (psoriasis non contagieux mais souvent perçu comme tel).

        ---
        ### 5. Options thérapeutiques (très synthétique)

        - **Topiques** : dermocorticoïdes, analogues de la vitamine D, combinaisons.  
        - **Photothérapie** (UVB, PUVA).  
        - **Systémiques conventionnels** : méthotrexate, ciclosporine, acitrétine, apremilast.  
        - **Biothérapies / molécules ciblées** : anti-TNFα, anti-IL-17, anti-IL-12/23, anti-IL-23, inhibiteurs de JAK, etc.  

        Le choix thérapeutique intègre :
        - la sévérité cutanée et articulaire,  
        - les comorbidités,  
        - les contraintes de suivi,  
        - le projet de vie du patient.

        ---
        Cette enquête France Psoriasis permet de mettre ces éléments en perspective à partir 
        du **vécu réel** des patients en France.
        """
    )

# ---------------------------------------------------------
# 6. DISCUSSION SCIENTIFIQUE
# ---------------------------------------------------------
elif page == "📝 Discussion scientifique":
    st.title("Discussion scientifique – Interprétation des résultats de l’enquête")

    n = len(data)
    ages = pd.to_numeric(data.get("xs2", pd.Series(dtype=float)), errors="coerce")
    age_mean = ages.mean()
    age_med = ages.median()

    sex_counts = data.get("s1", pd.Series(dtype=object)).value_counts()
    n_f = int(sex_counts.get("Une femme", 0))
    n_h = int(sex_counts.get("Un homme", 0))

    # Comorbidités (rs7aa si dispo sinon rs7ab)
    question_for_comor = "rs7aa" if "rs7aa" in question_to_cols else "rs7ab"
    comorbid_count = build_comorbidity_count(data, question_for_comor)

    top_comor = None
    age_by_comor = []
    if comorbid_count is not None:
        cols = question_to_cols.get(question_for_comor, [])
        rows = []
        for col in cols:
            ser = data[col].astype(str).str.strip()
            present = ser.notna() & (ser != "") & (ser != "0")
            count = present.sum()
            if count == 0:
                continue
            label = name_to_value.get(col) or name_to_desc.get(col) or col
            rows.append((label, count))

            ages_series = pd.to_numeric(data.get("xs2", pd.Series(dtype=float)), errors="coerce")
            age_yes = ages_series[present]
            age_no = ages_series[~present]
            if age_yes.count() >= 5 and age_no.count() >= 5:
                t_stat, p_val = stats.ttest_ind(
                    age_yes.dropna(),
                    age_no.dropna(),
                    equal_var=False,
                    nan_policy="omit",
                )
                age_by_comor.append(
                    {
                        "Comorbidité": label,
                        "Âge moyen Oui": age_yes.mean(),
                        "Âge moyen Non": age_no.mean(),
                        "p-value": p_val,
                    }
                )

        if rows:
            df_comor = pd.DataFrame(rows, columns=["Comorbidité", "Effectif"])
            df_comor["%"] = df_comor["Effectif"] / n * 100
            top_comor = df_comor.sort_values("Effectif", ascending=False).head(8)

    st.markdown(
        f"""
        ### 1. Profil général des répondants

        L’enquête France Psoriasis inclut **{n}** répondants.  
        L’âge moyen est d’environ **{age_mean:.1f} ans** (médiane ~ **{age_med:.1f} ans**), 
        ce qui correspond à une population adulte, avec une proportion importante de patients 
        d’âge mûr, susceptibles de cumuler plusieurs facteurs de risque cardiovasculaire.

        La répartition par sexe montre :
        - **{n_f} femmes**
        - **{n_h} hommes**

        Cette légère sur-représentation féminine est classique dans les enquêtes associatives 
        (engagement plus fréquent des femmes dans les démarches de santé).
        """
    )

    if top_comor is not None:
        st.markdown("### 2. Comorbidités les plus fréquemment rapportées")
        st.dataframe(top_comor)

        st.markdown(
            """
            Ces résultats objectivent une forte prévalence de comorbidités dans la population psoriasique,
            en particulier des composantes **métaboliques** et **rhumatologiques**.

            En pratique clinique, cela renforce :
            - la nécessité d’un **dépistage structuré** (poids, TA, bilan glycémique et lipidique) 
              lors des consultations de dermatologie ;  
            - l’importance de **questionner systématiquement** les patients sur les douleurs articulaires 
              et les symptômes anxio-dépressifs ;  
            - le besoin d’une **coordination étroite** avec le médecin traitant, les cardiologues, 
              endocrinologues et rhumatologues.
            """
        )

    if age_by_comor:
        df_age_comor = pd.DataFrame(age_by_comor).sort_values("p-value")
        st.markdown("### 3. Lien entre âge et comorbidités")
        st.dataframe(df_age_comor)

        st.markdown(
            """
            Dans plusieurs comorbidités, l’âge moyen des patients atteints apparaît plus élevé 
            que celui des patients non atteints.  
            Sans prétendre à une analyse causale, ces résultats s’inscrivent dans le concept de 
            **« marche psoriasique »**, où l’inflammation chronique de bas grade, associée aux 
            facteurs de style de vie, conduit progressivement à :

            - une augmentation de la masse grasse viscérale,  
            - une résistance à l’insuline,  
            - une dyslipidémie,  
            - puis à la survenue d’événements cardiovasculaires majeurs.
            """
        )

    # Exemple bivarié supplémentaire : sexe × comorbidités
    if comorbid_count is not None and "s1" in data.columns:
        st.markdown("### 4. Exemple d’analyse bivariée : sexe et présence de comorbidités")

        has_comor = comorbid_count > 0
        sex = data["s1"].fillna("Non renseigné")
        ct = pd.crosstab(sex, has_comor)
        chi2, p, dof, _ = stats.chi2_contingency(ct)

        st.markdown("Tableau croisé : sexe × au moins une comorbidité")
        st.dataframe(ct)

        st.markdown(f"- Chi² = **{chi2:.2f}**, p-value = **{p:.4f}**")

        st.markdown(
            """
            Ce type d’analyse permet de vérifier si les femmes rapportent plus souvent les comorbidités
            que les hommes (ou l’inverse), ce qui peut traduire à la fois des différences biologiques,
            mais aussi des **différences de recours aux soins** ou de perception de la maladie.
            """
        )

    # Exemple bivarié : région (si dispo) × présence de comorbidités
    if comorbid_count is not None and "qs3c" in data.columns:
        st.markdown("### 5. Exemple d’analyse bivariée : région et présence de comorbidités")

        has_comor = comorbid_count > 0
        region = data["qs3c"].fillna("Non renseigné")
        ct_reg = pd.crosstab(region, has_comor)
        chi2_reg, p_reg, dof_reg, _ = stats.chi2_contingency(ct_reg)

        st.markdown("Tableau croisé : région × au moins une comorbidité")
        st.dataframe(ct_reg)

        st.markdown(f"- Chi² = **{chi2_reg:.2f}**, p-value = **{p_reg:.4f}**")

        st.markdown(
            """
            Même si les effectifs par région sont parfois limités, ce type d’analyse peut faire émerger
            des **hypothèses régionales** (différences socio-économiques, accès différencié aux soins 
            spécialisés, densité de dermatologues, etc.) qui mériteraient d’être explorées dans 
            des travaux dédiés.
            """
        )

    st.markdown(
        """
        ---
        ### 6. Messages clés et recommandations

        1. **Dépistage systématique des comorbidités**  
           - Mesure de l’IMC, de la tension artérielle, bilan glucido-lipidique.  
           - Recherche de symptômes évocateurs de rhumatisme psoriasique et de troubles anxio-dépressifs.  

        2. **Prise en charge multidisciplinaire structurée**  
           - Coordination dermatologue–médecin traitant–rhumatologue–cardiologue–psychiatre/psychologue.  
           - Rôle central des équipes hospitalières et des réseaux de soins pour les formes sévères.  

        3. **Personnalisation des objectifs thérapeutiques**  
           - Intensification des traitements (biothérapies, petites molécules ciblées) chez les patients 
             comorbides ou à haut risque cardio-métabolique.  
           - Prise en compte de l’âge, du statut professionnel, des projets de grossesse, des préférences 
             du patient dans le choix des traitements.  

        4. **Rôle des associations de patients**  
           - Les données issues de France Psoriasis montrent la capacité de l’association à documenter 
             le vécu réel des patients et à produire des données **utiles à la santé publique**.  
           - Ces résultats peuvent nourrir des argumentaires pour améliorer l’**accès aux soins** 
             (consultations spécialisées, ETP, psychologues, etc.).

        ---
        ### 7. Perspectives de publication

        Les analyses proposées (descriptif détaillé, profils par comorbidités, exemples de tests bivariés)
        permettent de structurer un article du type :

        > **« Profil clinique, comorbidités et parcours de soins des patients atteints de psoriasis en France :
        résultats de l’enquête France Psoriasis »**

        L’application peut être utilisée pour générer directement :
        - le tableau des caractéristiques générales,  
        - le tableau des comorbidités,  
        - des tableaux de tests bivariés (âge / sexe / région / comorbidités),  
        qui serviront de base à la rédaction.
        """
    )

# ---------------------------------------------------------
# 7. HYPOTHÈSES & PRÉ-TRAITEMENT
# ---------------------------------------------------------
elif page == "📚 Hypothèses & pré-traitement":
    st.title("Hypothèses de pré-traitement et limites méthodologiques")

    st.markdown(
        """
        Cette section décrit de façon transparente les **choix de pré-traitement** et les **hypothèses**
        utilisés dans cette application pour analyser la base de données France Psoriasis.

        ### 1. Structure de la base

        - Onglet **`Codebook`** : dictionnaire des questions  
          - `Name` : identifiant de variable  
          - `Description` : texte de la question / de l’item  
          - `Value` : libellés des réponses possibles (pour les questions multi-éléments)  
        - Onglet **`Data`** : réponses individuelles des participants.

        ### 2. Regroupement au niveau des questions

        - Les variables de l’onglet Data sont initialement codées comme `Name`  
          (ex : `b6:1`, `b6:2`, `rs7ab_1:1`, etc.).  
        - Un identifiant de question est reconstruit par la fonction `get_question_id` :  
          - on supprime d’abord ce qui suit `:` (→ `b6:1` devient `b6`, `rs7ab_1:17` devient `rs7ab_1`),  
          - puis, s’il existe, ce qui suit `_` (→ `rs7ab_1` devient `rs7ab`).  
        - Toutes les colonnes partageant le même identifiant de question sont regroupées dans 
          `question_to_cols[question_id]`.  

        - Pour les identifiants listés par l’enquête (par ex. `rs7aa`, `rs7ab`, `rs7ba`, `rs7bb`, `a3`, `b5`, `b6`,
          `b10`, `recrs7aa`, `recrs7`, `recrs7a`), les questions sont **explicitement regroupées** par texte :
          - on extrait la partie de la Description située avant le `?`  
          - toutes les modalités associées (diabète, HTA, asthme, etc.) sont rassemblées sous la même question.  

        - Si une question comporte plusieurs colonnes binaires **0/1**, elle est traitée comme 
          **question à réponses multiples** (une seule entrée dans le selectbox, agrégation des items).  
        - Les questions ordinales de type **Likert** (Tout à fait d’accord, etc.) sont laissées 
          **item par item**.

        ### 3. Colonnes exclues des analyses

        - Préfixes techniques : `alerte`, `alertes`, `CW`, `CW_`, `CW_token`, `CW_status`,  
          `CW_firstdate`, `CW_firsttime`, `CW_finishdate`, `CW_finishtime`.  
        - Colonnes : `nbj`, `vague`, `fincontact`, `mode`, `revi`.  

        ### 4. Gestion des valeurs manquantes et aberrantes

        - Variables numériques :
          - tentative de conversion (`pd.to_numeric`),  
          - exclusion des variables avec trop peu de valeurs numériques ou faible variabilité,  
          - analyses réalisées uniquement sur les lignes non manquantes (pas d’imputation avancée).  
        - Variables catégorielles :
          - `0`, vide ou `NaN` → **« Non »**,  
          - `1` → **« Oui »**,  
          - autres modalités textuelles conservées.

        ### 5. Score de comorbidités

        - Pour les questions de type « maladies associées » (`rs7aa` / `rs7ab`),  
          un score de comorbidités global est calculé :  
          - chaque item > 0 (ou non vide) = comorbidité présente,  
          - le score est la **somme** des comorbidités présentes.  

        ### 6. Tests statistiques

        - **Chi²** : association entre deux variables catégorielles.  
        - **t-test de Student (Welch)** : comparaison de moyennes entre deux groupes indépendants.  
        - **V de Cramér** : mesure d’intensité de l’association entre deux variables catégorielles 
          (matrice présentée dans l’onglet *Analyses globales*).  
        - Seuil de significativité indicatif : **p < 0,05**, sans correction pour comparaisons multiples 
          (à discuter dans le manuscrit).

        ### 7. Clustering

        - Clustering K-means réalisé sur deux dimensions :
          - âge (xs2)  
          - nombre de comorbidités chroniques (score rs7aa/rs7ab)  
        - Standardisation non appliquée (ordre de grandeur comparable entre les deux variables).  
        - Interprétation descriptive, sans prétention à définir des sous-types « définitifs » de psoriasis.

        ### 8. Limites

        - Échantillon volontaire, non probabiliste → possible biais de sélection.  
        - Analyses transversales (pas de suivi longitudinal).  
        - Qualité des données dépendante du remplissage et de la compréhension du questionnaire.  
        - Pas d’ajustement multivarié systématique dans cette version (mais possible dans une version future).

        ### 9. Pistes d’évolution

        - Intégration de modèles multivariés (régressions logistiques, modèles linéaires).  
        - Meilleure exploitation des données de sévérité cutanée et articulaire (PASI, DLQI, scores de douleur).  
        - Exports automatiques des tableaux au format Excel / LaTeX pour faciliter la rédaction de publications.  
        - Développement de modules de simulation (par ex. impact d’une réduction de l’IMC sur le risque
          cardiovasculaire dans une population psoriasique).

        ---
        Ces hypothèses et choix méthodologiques peuvent être repris tels quels dans la 
        **section Matériels et Méthodes** de l’article.
        """
    )
