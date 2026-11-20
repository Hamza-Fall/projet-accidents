"""
Dashboard « Accidents corporels de la circulation en Île-de-France » (2022-2024).

Audience : responsable sécurité routière (préfecture, département) qui doit décider où, quand et vers quels usagers agir.
Source   : ONISR / Ministère de l'Intérieur, bases annuelles des accidents corporels (data.gouv.fr), licence ouverte.
Données  : base SQLite régénérée au démarrage à partir des CSV du dossier data/ (voir init_db.py).

En local :   python app.py    puis http://127.0.0.1:8050
Déploiement : gunicorn app:server --timeout 300
"""
import os
from functools import lru_cache

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

import db
import init_db

# La base n'est jamais versionnée : sur Render elle est reconstruite à CHAQUE démarrage (disque éphémère) ;
# en local, seulement si elle est absente (`python init_db.py` force la reconstruction).
if os.environ.get("RENDER") or not os.path.exists(db.DB_PATH):
    init_db.construire_base(afficher=False)

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
ANNEES = [2022, 2023, 2024]
DEPARTEMENTS = ["75", "77", "78", "91", "92", "93", "94", "95"]
NOMS_DEP = {"75": "75 – Paris", "77": "77 – Seine-et-Marne", "78": "78 – Yvelines", "91": "91 – Essonne",
            "92": "92 – Hauts-de-Seine", "93": "93 – Seine-Saint-Denis", "94": "94 – Val-de-Marne", "95": "95 – Val-d'Oise"}
MOIS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
JOURS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
ORDRE_GRAVITE = ["Tué", "Blessé hospitalisé", "Blessé léger", "Indemne"]
COULEUR_GRAVITE = {"Tué": "#7B1E1E", "Blessé hospitalisé": "#E4572E", "Blessé léger": "#F2B134", "Indemne": "#9DB4C0"}
COULEUR_ANNEE = {2022: "#9DB4C0", 2023: "#5B8DB8", 2024: "#1F4E79"}
BLEU = "#1F4E79"
SEUIL_POPULATION = 5000            # classement par habitant : communes de plus de 5 000 habitants (évite les petits nombres)
IDS_GRAPHES = ["courbe", "heatmap", "gravite", "communes", "nuage", "table"]


def fr(x, n=0):
    """Format français : 2 408 ou 2 407,9"""
    return f"{x:,.{n}f}".replace(",", " ").replace(".", ",")


# ----------------------------------------------------------------------------
# Données : requêtes SQL (via db.py), mises en cache par combinaison de filtres
# ----------------------------------------------------------------------------
@lru_cache(maxsize=48)
def donnees(annees, deps, graves, usager):
    return {
        "kpi": db.indicateurs(annees, deps, graves, usager),
        "mois": db.accidents_par_mois(annees, deps, graves, usager),
        "jh": db.accidents_jour_heure(annees, deps, graves, usager),
        "gravite": db.gravite_par_usager(annees, deps, graves, usager),
        "communes": db.communes_accidents(annees, deps, graves, usager),
        "points": db.points_noirs(annees, deps, graves, usager, limite=300),
    }


# Préchauffage : on calcule tout de suite les deux affichages les plus probables (le serveur gratuit est lent)
donnees(tuple(ANNEES), tuple(DEPARTEMENTS), False, "Tous")
donnees(tuple(ANNEES), tuple(DEPARTEMENTS), True, "Tous")


# ----------------------------------------------------------------------------
# Graphiques (le titre et le sous-titre sont affichés en HTML au-dessus du graphique : ils ne peuvent plus être coupés)
# ----------------------------------------------------------------------------
JOURS_LONG = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
COULEUR_TITRE = "#111827"
COULEUR_SOUS_TITRE = "#374151"
TAILLE_TEXTE = 14
TAILLE_AXES = 13


def style(fig, titre, legende=False, sous_titre=None):
    """Mise en forme commune. Le titre est mémorisé dans fig.layout.meta et affiché par entete() au-dessus du graphique."""
    fig.update_layout(template="plotly_white", autosize=True, height=None, showlegend=legende,
                      font=dict(size=TAILLE_TEXTE), hoverlabel=dict(font_size=TAILLE_TEXTE),
                      title=dict(text=""), meta=dict(titre=titre, sous=sous_titre or ""),
                      margin=dict(t=54 if legende else 20, l=70, r=28, b=56),
                      legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(size=TAILLE_TEXTE)))
    fig.update_xaxes(tickfont=dict(size=TAILLE_AXES), title_font=dict(size=TAILLE_TEXTE))
    fig.update_yaxes(tickfont=dict(size=TAILLE_AXES), title_font=dict(size=TAILLE_TEXTE))
    return fig


def entete(fig):
    """Titre-message (gras, 22 px) + contexte (15 px) à placer au-dessus du graphique."""
    meta = fig.layout.meta or {}
    titre, sous = meta.get("titre", ""), meta.get("sous", "")
    if not titre:
        return []
    enfants = [html.Div(titre, style={"fontSize": "22px", "fontWeight": "bold", "color": COULEUR_TITRE, "lineHeight": "1.25"})]
    if sous:
        enfants.append(html.Div(sous, style={"fontSize": "15px", "color": COULEUR_SOUS_TITRE, "marginTop": "4px"}))
    return enfants


def fig_vide(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color="#777"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, "")


def fig_courbe(mois_df, annees):                                 # courbe : évolution mensuelle
    fig = go.Figure()
    totaux = {}
    for a in annees:
        y = mois_df[mois_df["annee"] == a].set_index("mois")["accidents"].reindex(range(1, 13), fill_value=0)
        totaux[a] = int(y.sum())
        fig.add_trace(go.Scatter(x=MOIS, y=y.values, mode="lines+markers", name=str(a), line=dict(color=COULEUR_ANNEE[a], width=3),
                                 hovertemplate="%{x} " + str(a) + " : %{y} accidents<extra></extra>"))
    premiere, derniere = min(annees), max(annees)
    if len(annees) >= 2 and totaux[premiere] > 0:
        var = (totaux[derniere] / totaux[premiere] - 1) * 100
        titre = f"Accidents : {'−' if var < 0 else '+'}{fr(abs(var), 1)} % entre {premiere} et {derniere}"
    else:
        y = mois_df.groupby("mois")["accidents"].sum()
        titre = f"Mois le plus chargé : {MOIS[int(y.idxmax()) - 1]} ({fr(y.max())} accidents)" if len(y) else "Accidents par mois"
    style(fig, titre, legende=True, sous_titre="Accidents par mois, une courbe par année")
    fig.update_layout(yaxis=dict(title="Accidents", rangemode="tozero"))
    return fig


def fig_heatmap(jh_df):                                          # heatmap : jour × heure
    m = jh_df.pivot_table(index="jour_semaine", columns="heure", values="accidents", aggfunc="sum").reindex(index=range(7), columns=range(24)).fillna(0)
    jour, heure = np.unravel_index(np.argmax(m.values), m.shape)
    fig = go.Figure(go.Heatmap(z=m.values, x=list(range(24)), y=JOURS, colorscale="YlOrRd", colorbar=dict(thickness=16, tickfont=dict(size=TAILLE_AXES), title=dict(text="Acc.", side="top")),
                               hovertemplate="%{y} %{x}h : %{z} accidents<extra></extra>"))
    style(fig, f"Pic : le {JOURS_LONG[jour]} à {heure} h", sous_titre=f"{fr(m.values[jour, heure])} accidents · jour de la semaine × heure")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(dtick=3, title_text="Heure")
    return fig


def fig_gravite(g_df):                                           # barres empilées : gravité selon le type d'usager
    p = g_df[g_df["gravite"].isin(ORDRE_GRAVITE)].pivot_table(index="type_usager", columns="gravite", values="usagers", aggfunc="sum").fillna(0)
    p = p.reindex(columns=ORDRE_GRAVITE, fill_value=0)
    totaux = p.sum(axis=1)
    p = p[totaux >= 50]
    if p.empty:
        return fig_vide("Pas assez d'usagers pour ce filtre")
    part = p.div(p.sum(axis=1), axis=0) * 100
    touches = part["Tué"] + part["Blessé hospitalisé"]
    ordre = list(touches.sort_values().index)                    # le plus touché en haut
    fig = go.Figure()
    for g in ORDRE_GRAVITE:
        fig.add_trace(go.Bar(y=ordre, x=part.loc[ordre, g], orientation="h", name=g, marker_color=COULEUR_GRAVITE[g],
                             text=[f"{v:.0f} %" if v >= 8 else "" for v in part.loc[ordre, g]], textposition="inside",
                             textfont=dict(size=TAILLE_AXES, color="white" if g in ("Tué", "Blessé hospitalisé") else "#1F2937"),
                             hovertemplate="%{y} – " + g + " : %{x:.1f} %<extra></extra>"))
    pire = touches.idxmax()
    style(fig, f"{pire} : {fr(touches.max())} % tués ou hospitalisés", legende=True, sous_titre="Gravité des usagers selon leur type (en %)")
    fig.update_layout(barmode="stack", xaxis=dict(ticksuffix=" %", range=[0, 100]), margin=dict(l=180))
    return fig


def fig_communes(c_df):                                          # barres horizontales : classement des communes
    c = c_df.copy()
    par_habitant = c["population"].notna().any()
    if par_habitant:
        c = c[c["population"] >= SEUIL_POPULATION].copy()
        c["valeur"] = c["accidents_graves"] / c["population"] * 100000
        unite, sous = "acc. graves / 100 000 hab.", f"10 premières communes de plus de {fr(SEUIL_POPULATION)} habitants"
    else:
        c["valeur"] = c["accidents_graves"]
        unite, sous = "accidents graves", "Population non chargée : classement en nombre brut"
    c = c.sort_values("valeur", ascending=False).head(10)
    if c.empty:
        return fig_vide("Aucune commune pour ce filtre")
    c = c.iloc[::-1]                                             # la plus touchée en haut
    c["commune"] = c["commune"].map(lambda n: n if len(n) <= 26 else n[:25] + "…")
    fig = go.Figure(go.Bar(y=c["commune"], x=c["valeur"], orientation="h", marker_color=BLEU,
                           text=[fr(v, 1 if par_habitant else 0) for v in c["valeur"]], textposition="outside", textfont=dict(size=TAILLE_AXES), cliponaxis=False,
                           customdata=c[["accidents", "accidents_graves"]].values,
                           hovertemplate="%{y} : %{customdata[0]} accidents dont %{customdata[1]} graves<extra></extra>"))
    top = c.iloc[-1]
    style(fig, f"{top['commune']} : {fr(top['valeur'], 1 if par_habitant else 0)} {unite}", sous_titre=sous)
    fig.update_layout(margin=dict(l=200, r=60), xaxis=dict(visible=False))
    return fig


def fig_nuage(c_df):                                             # nuage de points : accidents contre part de graves
    c = c_df[c_df["accidents"] >= 10].copy()
    if len(c) < 3:
        return fig_vide("Pas assez de communes (au moins 10 accidents) pour ce filtre")
    c["pct"] = c["accidents_graves"] / c["accidents"] * 100
    moyenne = c_df["accidents_graves"].sum() / c_df["accidents"].sum() * 100
    gros = c[c["accidents"] >= 30]
    if len(gros):
        pire = gros.loc[gros["pct"].idxmax()]
        titre = f"À {pire['commune']}, {fr(pire['pct'])} % des accidents sont graves"
    else:
        titre = f"{fr(moyenne)} % des accidents sont graves en moyenne"
    fig = go.Figure(go.Scatter(x=c["accidents"], y=c["pct"], mode="markers", text=c["commune"],
                               marker=dict(size=10, color=BLEU, opacity=0.6, line=dict(width=0.5, color="white")),
                               hovertemplate="<b>%{text}</b><br>%{x} accidents<br>%{y:.1f} % graves<extra></extra>"))
    fig.add_hline(y=moyenne, line_dash="dot", line_color="#999")
    style(fig, titre, sous_titre=f"Moyenne régionale : {fr(moyenne)} % · communes de 10 accidents ou plus")
    fig.update_layout(xaxis=dict(title="Accidents (échelle log)", type="log", tickvals=[10, 30, 100, 300, 1000], ticktext=["10", "30", "100", "300", "1 000"]), yaxis=dict(title="Accidents graves (%)", ticksuffix=" %", rangemode="tozero"))
    return fig


def fig_table(p_df):                                             # tableau : lieux les plus graves
    t = p_df.head(8)
    if t.empty:
        return fig_vide("Aucun lieu géolocalisé pour ce filtre")
    lieu = t["lieu"].fillna("—").map(lambda n: n if len(n) <= 26 else n[:25] + "…")
    commune = t["commune"].map(lambda n: n if len(n) <= 18 else n[:17] + "…")
    fig = go.Figure(go.Table(
        columnwidth=[3, 2.2, 1.2, 1.2, 1.1],
        header=dict(values=["<b>Lieu</b>", "<b>Commune</b>", "<b>Accid.</b>", "<b>Graves</b>", "<b>Tués</b>"],
                    fill_color=BLEU, font=dict(color="white", size=TAILLE_TEXTE), align="left", height=32),
        cells=dict(values=[lieu, commune, t["accidents"], t["graves"], t["tues"]],
                   fill_color="#F5F6F8", align="left", font=dict(size=TAILLE_TEXTE), height=30)))
    style(fig, f"8 lieux : {fr(t['graves'].sum())} accidents graves", sous_titre="Lieux regroupés à ~100 m, classés par gravité")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ----------------------------------------------------------------------------
# Carte Folium : points noirs (lieux regroupés à ~100 m), colorés selon le nombre d'accidents graves
# ----------------------------------------------------------------------------
def couleur_points(graves):
    return "#7B1E1E" if graves >= 4 else "#E4572E" if graves >= 2 else "#F2B134" if graves == 1 else "#9DB4C0"


LEGENDE_CARTE = """
<div style="position:fixed; bottom:14px; left:10px; z-index:9999; background:white; padding:8px 12px; border:1px solid #999;
            border-radius:6px; font-family:Arial; font-size:13px; box-shadow:0 1px 4px rgba(0,0,0,.3)">
  <b>Points noirs</b> (lieux regroupés à ~100 m, 300 plus graves)<br>
  <span style="color:#7B1E1E">●</span> 4 accidents graves ou plus &nbsp;
  <span style="color:#E4572E">●</span> 2 à 3 &nbsp;
  <span style="color:#F2B134">●</span> 1 &nbsp;
  <span style="color:#9DB4C0">●</span> aucun<br>
  Taille du cercle = nombre d'accidents
</div>"""

CARTE_VIDE = ("<html><body style='font-family:Arial;color:#777;display:flex;align-items:center;justify-content:center;height:100%'>"
              "<p>Aucun lieu géolocalisé pour ce filtre.</p></body></html>")


def carte_html(p_df):
    if p_df.empty:
        return CARTE_VIDE
    m = folium.Map(location=[48.85, 2.35], zoom_start=10, tiles=None, control_scale=True)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles © Esri", name="Fond clair (Esri)", max_zoom=16).add_to(m)
    m.fit_bounds([[p_df["lat"].min(), p_df["lon"].min()], [p_df["lat"].max(), p_df["lon"].max()]], padding=(30, 30))
    for r in p_df.itertuples():
        popup = (f"<div style='font-family:Arial;font-size:13px'><b>{r.lieu if isinstance(r.lieu, str) else 'Lieu non renseigné'}</b><br>{r.commune}"
                 f"<br>{r.accidents} accident(s) dont <b>{r.graves}</b> grave(s), {int(r.tues)} avec tué(s)</div>")
        folium.CircleMarker(location=[r.lat, r.lon], radius=min(4 + 2.5 * np.sqrt(r.accidents), 22), color="white", weight=1,
                            fill=True, fill_color=couleur_points(r.graves), fill_opacity=0.85,
                            tooltip=f"{r.commune} – {r.accidents} accidents, {r.graves} graves", popup=folium.Popup(popup, max_width=260)).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGENDE_CARTE))
    return m.get_root().render()


# ----------------------------------------------------------------------------
# Mise en page : la page défile normalement du haut vers le bas
# ----------------------------------------------------------------------------
def cartes_kpi(k):
    taux = k["tues"] / k["accidents"] * 100 if k["accidents"] else 0
    cartes = [("Accidents corporels", fr(k["accidents"]), "#1F4E79"), ("Tués", fr(k["tues"]), "#7B1E1E"),
              ("Blessés hospitalisés", fr(k["hospitalises"]), "#E4572E"), ("Tués pour 100 accidents", fr(taux, 1), "#555")]
    return [html.Div(style={"flex": "1 1 0", "background": "white", "border": "1px solid #E0E0E0", "borderRadius": "8px",
                            "padding": "10px 16px", "borderLeft": f"6px solid {couleur}"},
                     children=[html.Div(titre, style={"fontSize": "15px", "color": "#4B5563"}),
                               html.Div(valeur, style={"fontSize": "32px", "fontWeight": "bold", "color": couleur})])
            for titre, valeur, couleur in cartes]


app = Dash(__name__)
app.title = "Accidents de la route en Île-de-France"
server = app.server                      # exposé pour gunicorn : gunicorn app:server
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>html, body {margin: 0; min-height: 100%; overflow-y: auto;}</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>"""

PANNEAU = {"position": "relative", "backgroundColor": "white", "border": "1px solid #E0E0E0", "borderRadius": "8px",
           "minHeight": 0, "minWidth": 0, "overflow": "hidden"}
PANNEAU_GRAPHE = {**PANNEAU, "display": "flex", "flexDirection": "column"}
REMPLIR = {"position": "absolute", "top": "6px", "left": "6px", "right": "6px", "bottom": "6px"}      # marge intérieure
CONFIG = {"displayModeBar": False, "responsive": True}
ETIQUETTE = {"fontWeight": "bold", "fontSize": "13px", "color": "#374151", "display": "block", "marginBottom": "3px"}
HAUTEUR = "580px"                                                # hauteur fixe d'un panneau : les textes gardent leur taille
ONGLET = {"padding": "12px 20px", "fontWeight": "bold", "fontSize": "15px", "color": "#374151", "backgroundColor": "#E5E7EB"}
ONGLET_ACTIF = {**ONGLET, "color": BLEU, "backgroundColor": "white", "borderTop": f"3px solid {BLEU}"}


def graphique(nom, hauteur=HAUTEUR):
    """Un panneau = titre HTML (rempli par le callback) + graphique en dessous."""
    return html.Div(style={**PANNEAU_GRAPHE, "height": hauteur}, children=[
        html.Div(id=f"t-{nom}", style={"flex": "0 0 auto", "padding": "14px 18px 0 18px"}),
        html.Div(style={"position": "relative", "flex": "1 1 auto", "minHeight": 0},
                 children=dcc.Graph(id=f"g-{nom}", style=REMPLIR, config=CONFIG)),
    ])


def grille(colonnes, enfants, lignes=None):
    style_grille = {"display": "grid", "gap": "16px", "gridTemplateColumns": colonnes}
    if lignes:
        style_grille["gridTemplateRows"] = lignes
    return html.Div(style=style_grille, children=enfants)


def filtre(titre, largeur, composant):
    return html.Div(style={"width": largeur}, children=[html.Label(titre, style=ETIQUETTE), composant])


app.layout = html.Div(
    style={"minHeight": "100vh", "boxSizing": "border-box", "display": "flex", "flexDirection": "column", "gap": "12px",
           "padding": "12px 18px 30px 18px", "fontFamily": "Arial, sans-serif", "backgroundColor": "#F5F6F8"},
    children=[
        # Bandeau : titre + filtres (un seul jeu de filtres pour tous les visuels)
        html.Div(
            style={"display": "flex", "alignItems": "flex-end", "justifyContent": "space-between", "gap": "16px",
                   "flexWrap": "wrap", "flex": "0 0 auto", "position": "relative", "zIndex": 10},
            children=[
                html.Div([
                    html.H1("Accidents de la route en Île-de-France : où, quand et chez qui agir ?",
                            style={"margin": 0, "fontSize": "26px", "color": BLEU}),
                    html.Div("Accidents corporels 2022-2024 · Source : ONISR / Ministère de l'Intérieur (data.gouv.fr)",
                             style={"fontSize": "14px", "color": "#4B5563", "marginTop": "4px"}),
                ]),
                html.Div(style={"display": "flex", "gap": "12px", "alignItems": "flex-end", "flexWrap": "wrap"}, children=[
                    filtre("Année(s)", "150px", dcc.Dropdown(id="filtre-annee", options=[{"label": str(a), "value": a} for a in ANNEES],
                                                             value=ANNEES, multi=True, clearable=False)),
                    filtre("Département(s)", "240px", dcc.Dropdown(id="filtre-dep", options=[{"label": NOMS_DEP[d], "value": d} for d in DEPARTEMENTS],
                                                                   value=DEPARTEMENTS, multi=True, clearable=False)),
                    filtre("Accidents affichés", "200px", dcc.Dropdown(
                        id="filtre-gravite", options=[{"label": "Tous les accidents", "value": "tous"},
                                                      {"label": "Accidents graves (tué ou hospitalisé)", "value": "graves"}],
                        value="tous", clearable=False)),
                    filtre("Usager impliqué", "200px", dcc.Dropdown(
                        id="filtre-usager", options=[{"label": "Tous les usagers", "value": "Tous"}] + [{"label": t, "value": t} for t in db.TYPES_USAGERS],
                        value="Tous", clearable=False)),
                ]),
            ]),

        # Chiffres clés
        html.Div(id="kpi", style={"display": "flex", "gap": "12px", "flex": "0 0 auto"}),

        # Trois onglets : chacun montre peu de visuels, en grand
        dcc.Tabs(
            id="onglets", value="quand", style={"height": "50px", "flex": "0 0 auto"}, content_style={"paddingTop": "14px"},
            colors={"primary": BLEU, "background": "#E5E7EB", "border": "#D1D5DB"},
            children=[
                dcc.Tab(label="Quand ?  Évolution et créneaux", value="quand", style=ONGLET, selected_style=ONGLET_ACTIF,
                        children=grille("minmax(0, 1fr) minmax(0, 1fr)", [graphique("courbe"), graphique("heatmap")])),
                dcc.Tab(label="Qui ?  Usagers et gravité", value="qui", style=ONGLET, selected_style=ONGLET_ACTIF,
                        children=grille("minmax(0, 1fr) minmax(0, 1fr)", [graphique("gravite"), graphique("nuage")])),
                dcc.Tab(label="Où ?  Communes et points noirs", value="ou", style=ONGLET, selected_style=ONGLET_ACTIF,
                        children=grille("minmax(0, 1.25fr) minmax(0, 1fr)", [
                            html.Div(style={**PANNEAU, "gridRow": "1 / span 2"}, children=html.Iframe(
                                id="carte", srcDoc="", style={**REMPLIR, "width": "calc(100% - 12px)", "height": "calc(100% - 12px)", "border": "none"})),
                            graphique("communes", hauteur="470px"), graphique("table", hauteur="470px")],
                            lignes="470px 470px")),
            ]),
    ])


# ----------------------------------------------------------------------------
# Callback : un seul jeu de filtres -> 6 graphiques + 6 titres + chiffres clés + carte
# ----------------------------------------------------------------------------
SORTIES = ([Output("kpi", "children")]
           + [sortie for nom in IDS_GRAPHES for sortie in (Output(f"g-{nom}", "figure"), Output(f"t-{nom}", "children"))]
           + [Output("carte", "srcDoc")])


def assembler(kpi, figs, carte):
    """Range les résultats dans l'ordre des Output : kpi, (figure, titre) pour chaque graphique, carte."""
    resultat = [kpi]
    for nom in IDS_GRAPHES:
        resultat += [figs[nom], entete(figs[nom])]
    resultat.append(carte)
    return tuple(resultat)


@app.callback(*SORTIES,
              Input("filtre-annee", "value"), Input("filtre-dep", "value"),
              Input("filtre-gravite", "value"), Input("filtre-usager", "value"))
def mettre_a_jour(annees, deps, gravite, usager):
    annees = tuple(sorted(a for a in (annees or []) if a in ANNEES))
    deps = tuple(sorted(d for d in (deps or []) if d in DEPARTEMENTS))
    if not annees or not deps:
        vide = fig_vide("Sélectionnez au moins une année et un département")
        return assembler(cartes_kpi({"accidents": 0, "tues": 0, "hospitalises": 0}), {n: vide for n in IDS_GRAPHES}, CARTE_VIDE)
    d = donnees(annees, deps, gravite == "graves", usager or "Tous")
    if d["kpi"]["accidents"] == 0:
        vide = fig_vide("Aucun accident pour ces filtres")
        return assembler(cartes_kpi(d["kpi"]), {n: vide for n in IDS_GRAPHES}, CARTE_VIDE)
    figs = {"courbe": fig_courbe(d["mois"], list(annees)), "heatmap": fig_heatmap(d["jh"]), "gravite": fig_gravite(d["gravite"]),
            "communes": fig_communes(d["communes"]), "nuage": fig_nuage(d["communes"]), "table": fig_table(d["points"])}
    return assembler(cartes_kpi(d["kpi"]), figs, carte_html(d["points"]))


if __name__ == "__main__":
    app.run(debug=True)