import flet as ft
import json
import math
import os
import csv
from datetime import datetime


# ==========================================
# 1. BACKEND: LOGIC & MODEL
# ==========================================

class PredictionModel:
    def __init__(self):
        self.params = self.load_params()

    def load_params(self):
        if os.path.exists("model_params.json"):
            try:
                with open("model_params.json", "r") as f:
                    return json.load(f)
            except:
                pass
        return {
            "means": [20, 3.0, 3, 3, 3, 1, 0.5, 0.5],
            "stds": [5, 0.5, 1, 1, 1, 1, 0.5, 0.5],
            "coefficients": [0.1, -0.5, 0.8, -0.4, 0.6, -0.3, 0.1, 0.5],
            "intercept": -1.5
        }

    def _convert_input(self, val, type_func, default):
        try:
            return type_func(val)
        except:
            return default

    def predict_proba(self, inputs):
        age = self._convert_input(inputs.get('age'), float, 20.0)
        cgpa = self._convert_input(inputs.get('cgpa'), float, 3.0)
        ap = self._convert_input(inputs.get('academic_pressure'), float, 3.0)
        ss = self._convert_input(inputs.get('study_satisfaction'), float, 3.0)
        fs = self._convert_input(inputs.get('financial_stress'), float, 3.0)
        g_val = inputs.get('gender', 0)
        gender = 1.0 if isinstance(g_val, str) and g_val.lower() in ["nam", "male"] else (
            float(g_val) if not isinstance(g_val, str) else 0.0)
        fh_val = inputs.get('family_history', 0)
        fh = 1.0 if isinstance(fh_val, str) and fh_val.lower() in ["có", "yes", "1"] else (
            float(fh_val) if not isinstance(fh_val, str) else 0.0)
        s_val = inputs.get('sleep_duration', 1.0)
        if isinstance(s_val, str):
            s_str = s_val.lower()
            if "dưới 5" in s_str:
                sd = 0.0
            elif "5-6" in s_str:
                sd = 1.0
            elif "7-8" in s_str:
                sd = 2.0
            elif "trên 8" in s_str:
                sd = 3.0
            else:
                sd = 1.0
        else:
            sd = float(s_val)

        raw_features = [age, cgpa, ap, ss, fs, sd, gender, fh]
        z_scores = []
        for i, val in enumerate(raw_features):
            mean = self.params['means'][i]
            std = self.params['stds'][i]
            if std == 0: std = 1
            z_scores.append((val - mean) / std)

        logit = self.params['intercept']
        for i, z in enumerate(z_scores):
            logit += z * self.params['coefficients'][i]

        return 1.0 / (1.0 + math.exp(-logit))

    def predict(self, inputs):
        prob = self.predict_proba(inputs)
        ap = float(inputs['academic_pressure'])
        ss = float(inputs['study_satisfaction'])
        fs = float(inputs['financial_stress'])
        s_val = inputs['sleep_duration']
        sd_map = {"Dưới 5 tiếng": 0.0, "5-6 tiếng": 1.0, "7-8 tiếng": 2.0, "Trên 8 tiếng": 3.0}
        sd = sd_map.get(s_val, 1.0)

        risk_factors = {
            "Áp lực học": ap / 5.0,
            "Tài chính": fs / 5.0,
            "Thiếu ngủ": 1.0 if sd < 1.0 else (0.5 if sd < 2.0 else 0.1),
            "Chán học": 1.0 - (ss / 5.0)
        }
        return prob, risk_factors


# ==========================================
# 2. FRONTEND: UI
# ==========================================

def main(page: ft.Page):
    page.title = "Mental Health AI Pro"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.window_width = 1350
    page.window_height = 900  # Tăng chiều cao lên chút cho thoáng
    page.theme = ft.Theme(page_transitions=ft.PageTransitionsTheme(windows=ft.PageTransitionTheme.CUPERTINO))

    model = PredictionModel()
    history_data = []

    class AppColors:
        LightBG = ft.Colors.BLUE_GREY_50
        DarkBG = ft.Colors.GREY_900
        LightCard = ft.Colors.WHITE
        DarkCard = ft.Colors.GREY_800
        Primary = ft.Colors.INDIGO_600
        Secondary = ft.Colors.PINK_500
        Success = ft.Colors.TEAL_600
        Warning = ft.Colors.AMBER_600
        Danger = ft.Colors.RED_600

    def create_fancy_icon(icon_name, color, bg_color):
        return ft.Container(
            content=ft.Icon(icon_name, color=color, size=30),
            bgcolor=bg_color, padding=12, border_radius=14,
            shadow=ft.BoxShadow(blur_radius=10, color=ft.Colors.with_opacity(0.3, bg_color))
        )

    # --- INPUTS ---
    txt_age = ft.TextField(label="Tuổi", value="20", expand=True, border_radius=12, text_size=15,
                           prefix_icon=ft.Icons.PERSON)
    dd_gender = ft.Dropdown(label="Giới tính", expand=True, border_radius=12, text_size=15, prefix=ft.Icon(ft.Icons.WC),
                            options=[ft.dropdown.Option("Nam"), ft.dropdown.Option("Nữ")], value="Nữ")
    txt_cgpa = ft.TextField(label="CGPA (0-4.0)", value="3.5", expand=True, border_radius=12, text_size=15,
                            prefix_icon=ft.Icons.SCHOOL)

    sl_ap = ft.Slider(min=1, max=5, divisions=4, label="{value}", value=3, active_color=AppColors.Primary)
    sl_ss = ft.Slider(min=1, max=5, divisions=4, label="{value}", value=3, active_color=AppColors.Success)
    sl_fs = ft.Slider(min=1, max=5, divisions=4, label="{value}", value=3, active_color=AppColors.Warning)

    dd_sleep = ft.Dropdown(label="Giấc ngủ trung bình", width=220, border_radius=12, text_size=15,
                           prefix=ft.Icon(ft.Icons.BEDTIME),
                           options=[ft.dropdown.Option("Dưới 5 tiếng"), ft.dropdown.Option("5-6 tiếng"),
                                    ft.dropdown.Option("7-8 tiếng"), ft.dropdown.Option("Trên 8 tiếng")],
                           value="7-8 tiếng")
    dd_history = ft.Dropdown(label="Tiền sử gia đình?", expand=True, border_radius=12, text_size=15,
                             prefix=ft.Icon(ft.Icons.FAMILY_RESTROOM),
                             options=[ft.dropdown.Option("Có"), ft.dropdown.Option("Không")], value="Không")

    # --- RESULTS CONTROLS (BIGGER SIZE) ---

    # 1. Icon & Text kết quả (To hơn)
    res_icon = ft.Icon(ft.Icons.ANALYTICS_ROUNDED, size=70, color=ft.Colors.INDIGO_200)
    res_text = ft.Text("...", size=36, weight=ft.FontWeight.W_900)

    # 2. Thanh ProgressBar (Rộng hơn và dày hơn)
    res_bar = ft.ProgressBar(width=650, height=18, border_radius=9, value=0)

    # 3. Lời khuyên (Khung to hơn, chữ to hơn)
    txt_advice = ft.Text("Vui lòng nhập thông tin để nhận tư vấn.", size=16, text_align=ft.TextAlign.CENTER,
                         color=ft.Colors.BLUE_GREY_700)
    advice_box = ft.Container(
        content=txt_advice,
        padding=20,  # Padding dày hơn
        bgcolor=ft.Colors.BLUE_50,
        border_radius=12,
        width=700,  # Rộng hơn hẳn
        alignment=ft.alignment.center
    )

    # 4. Grid Container (Chứa các thanh to hơn)
    risk_grid = ft.Column(spacing=15)

    result_container = ft.Column(
        [
            res_icon,
            res_text,
            res_bar,
            ft.Container(height=10),
            advice_box,
            ft.Divider(height=40, thickness=1),
            ft.Text("CHI TIẾT CÁC YẾU TỐ RỦI RO", weight="bold", size=14, color=ft.Colors.GREY_600),
            risk_grid
        ],
        opacity=0, offset=ft.Offset(0, 0.1),
        animate_opacity=ft.Animation(800, "easeOut"), animate_offset=ft.Animation(800, "easeOutBack"),
        alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=5
    )

    # --- HISTORY ---
    history_table = ft.DataTable(
        width=float("inf"),
        columns=[ft.DataColumn(ft.Text("Thời gian", weight="bold")), ft.DataColumn(ft.Text("Kết quả", weight="bold")),
                 ft.DataColumn(ft.Text("Mức độ nguy cơ", weight="bold", text_align=ft.TextAlign.RIGHT), numeric=True)],
        rows=[], border_radius=10, heading_row_color=ft.Colors.with_opacity(0.1, AppColors.Primary),
        data_row_max_height=60,
        vertical_lines=ft.border.BorderSide(1, ft.Colors.GREY_100),
        horizontal_lines=ft.border.BorderSide(1, ft.Colors.GREY_100),
    )

    # --- FUNCTIONS ---

    def show_success_dialog(filepath):
        dlg = ft.AlertDialog(
            title=ft.Text("Xuất file thành công!", weight="bold"),
            content=ft.Column([
                ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN, size=60),
                ft.Text(f"Lưu tại:\n{filepath}", text_align=ft.TextAlign.CENTER, size=14),
            ], height=160, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            actions=[ft.TextButton("Đóng", on_click=lambda e: page.close(dlg))],
            actions_alignment=ft.MainAxisAlignment.CENTER
        )
        page.open(dlg)

    def generate_advice(prob, risk_factors):
        if prob >= 0.5:
            base_advice = "⚠️ CẢNH BÁO: Mức độ rủi ro cao. Bạn nên tìm kiếm sự hỗ trợ."
        else:
            base_advice = "✅ TỐT: Trạng thái ổn định. Hãy duy trì lối sống hiện tại."

        max_risk = max(risk_factors, key=risk_factors.get)
        if risk_factors[max_risk] >= 0.6:
            if max_risk == "Áp lực học":
                detail = "\n(Chú ý: Giảm tải áp lực học tập, chia nhỏ thời gian học)"
            elif max_risk == "Tài chính":
                detail = "\n(Chú ý: Lập kế hoạch chi tiêu hợp lý hơn)"
            elif max_risk == "Thiếu ngủ":
                detail = "\n(Chú ý: Cải thiện giấc ngủ, ngủ đủ 7-8 tiếng)"
            elif max_risk == "Chán học":
                detail = "\n(Chú ý: Tìm kiếm cảm hứng mới trong học tập)"
            else:
                detail = ""
            return base_advice + detail
        return base_advice

    def create_risk_bar_expanded(label, value):
        """Tạo thanh risk to và rộng hơn"""
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text(label, size=13, weight="bold"),  # Chữ to hơn
                    ft.Text(f"{int(value * 100)}%", size=13, weight="bold", color=ft.Colors.GREY_600)
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.ProgressBar(
                    value=value, height=12, border_radius=6,  # Thanh dày hơn (12px)
                    color=ft.Colors.ORANGE_400 if value > 0.6 else ft.Colors.BLUE_300,
                    bgcolor=ft.Colors.GREY_100
                )
            ], spacing=5),
            width=320,  # Tăng chiều rộng lên 320
            padding=5
        )

    def run_prediction(e):
        try:
            inputs = {'age': txt_age.value, 'cgpa': txt_cgpa.value, 'gender': dd_gender.value,
                      'academic_pressure': sl_ap.value, 'study_satisfaction': sl_ss.value,
                      'financial_stress': sl_fs.value, 'sleep_duration': dd_sleep.value,
                      'family_history': dd_history.value}
            prob, risk_factors = model.predict(inputs)
            percent = int(prob * 100)

            if prob >= 0.5:
                status = "NGUY CƠ CAO"
                color = AppColors.Danger
                icon_res = ft.Icons.WARNING_AMBER_ROUNDED
                bg_advice = ft.Colors.RED_50
                txt_col_advice = ft.Colors.RED_900
            else:
                status = "AN TOÀN"
                color = AppColors.Success
                icon_res = ft.Icons.VERIFIED_USER_ROUNDED
                bg_advice = ft.Colors.GREEN_50
                txt_col_advice = ft.Colors.GREEN_900

            res_text.value = f"{status} ({percent}%)"
            res_text.color = color

            txt_advice.value = generate_advice(prob, risk_factors)
            txt_advice.color = txt_col_advice
            advice_box.bgcolor = bg_advice

            res_bar.value = prob
            res_bar.color = color

            result_container.controls[0] = create_fancy_icon(icon_res, ft.Colors.WHITE, color)

            # --- GRID 2x2 LỚN HƠN ---
            risk_grid.controls.clear()
            items = []
            for k, v in risk_factors.items():
                items.append(create_risk_bar_expanded(k, v))

            if len(items) >= 4:
                risk_grid.controls.append(ft.Row([items[0], items[1]], alignment=ft.MainAxisAlignment.SPACE_BETWEEN))
                risk_grid.controls.append(ft.Row([items[2], items[3]], alignment=ft.MainAxisAlignment.SPACE_BETWEEN))
            else:
                for item in items: risk_grid.controls.append(item)

            result_container.opacity = 1
            result_container.offset = ft.Offset(0, 0)

            ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            history_data.append({"time": ts, "input": inputs, "status": status, "prob": f"{prob:.2%}"})
            history_table.rows.insert(0, ft.DataRow(cells=[
                ft.DataCell(ft.Text(ts)),
                ft.DataCell(
                    ft.Container(content=ft.Text(status, size=11, color=ft.Colors.WHITE, weight="bold"), bgcolor=color,
                                 padding=ft.padding.symmetric(horizontal=10, vertical=5), border_radius=20)),
                ft.DataCell(ft.Text(f"{prob:.1%}", weight="bold"))
            ]))
            page.update()
        except ValueError:
            page.snack_bar = ft.SnackBar(ft.Text("Lỗi nhập liệu!"), bgcolor=ft.Colors.RED);
            page.snack_bar.open = True;
            page.update()

    def export_csv(e):
        if not history_data:
            page.snack_bar = ft.SnackBar(ft.Text("Chưa có dữ liệu!"), bgcolor=ft.Colors.RED);
            page.snack_bar.open = True;
            page.update()
            return
        fn = f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(fn, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "Age", "Gender", "Result", "Probability"])
                for i in history_data: writer.writerow(
                    [i['time'], i['input']['age'], i['input']['gender'], i['status'], i['prob']])
            show_success_dialog(os.path.abspath(fn))
        except Exception as ex:
            page.snack_bar = ft.SnackBar(ft.Text(f"Lỗi: {str(ex)}"), bgcolor=ft.Colors.RED);
            page.snack_bar.open = True;
            page.update()

    # --- LAYOUT ---

    header = ft.Container(
        content=ft.Row([
            create_fancy_icon(ft.Icons.PSYCHOLOGY, AppColors.Primary, ft.Colors.WHITE),
            ft.Text("AI Mental Health Monitor", color=ft.Colors.WHITE, size=24, weight="bold"),
            ft.Container(expand=True),
        ]),
        padding=ft.padding.symmetric(horizontal=30, vertical=20),
        gradient=ft.LinearGradient(colors=[AppColors.Primary, ft.Colors.PURPLE_600]),
        shadow=ft.BoxShadow(blur_radius=15, color=ft.Colors.with_opacity(0.4, AppColors.Primary))
    )

    left_content = ft.Column([
        ft.Row([create_fancy_icon(ft.Icons.EDIT_NOTE, ft.Colors.WHITE, AppColors.Primary),
                ft.Text("NHẬP THÔNG TIN", size=18, weight="bold", color=ft.Colors.INDIGO_400)], spacing=15),
        ft.Divider(color=ft.Colors.TRANSPARENT, height=5),
        ft.Text("Thông tin cơ bản", weight="bold"),
        ft.Row([txt_age, dd_gender], spacing=15),
        ft.Row([txt_cgpa, dd_history], spacing=15),
        ft.Container(height=10),
        ft.Text("Đánh giá tâm lý & Lối sống", weight="bold"),
        ft.Column([
            ft.Row(
                [ft.Icon(ft.Icons.PSYCHOLOGY_ALT, size=16, color=AppColors.Primary), ft.Text("Áp lực học tập", size=13),
                 ft.Container(expand=True), sl_ap], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([ft.Icon(ft.Icons.SENTIMENT_SATISFIED_ALT, size=16, color=AppColors.Success),
                    ft.Text("Hài lòng việc học", size=13), ft.Container(expand=True), sl_ss],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row(
                [ft.Icon(ft.Icons.ATTACH_MONEY, size=16, color=AppColors.Warning), ft.Text("Áp lực tài chính", size=13),
                 ft.Container(expand=True), sl_fs], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ], spacing=0),
        ft.Container(height=5), dd_sleep,
        ft.Container(expand=True),
        ft.ElevatedButton("PHÂN TÍCH NGAY", on_click=run_prediction, icon=ft.Icons.ROCKET_LAUNCH,
                          style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12), bgcolor=AppColors.Primary,
                                               color=ft.Colors.WHITE, padding=25), width=600,
                          animate_scale=ft.Animation(200, "bounceOut"))
    ], spacing=10, expand=True)

    # Result Tab (Tăng padding lên cho cân đối)
    tab_result_content = ft.Container(
        content=result_container,
        padding=30,
        alignment=ft.alignment.center
    )

    tab_history = ft.Column([
        ft.Row([ft.Text("Lịch sử quét", size=20, weight="bold"),
                ft.ElevatedButton("Xuất file CSV", icon=ft.Icons.DOWNLOAD, on_click=export_csv,
                                  bgcolor=ft.Colors.GREEN_600, color=ft.Colors.WHITE)],
               alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ft.Container(height=10),
        ft.Container(content=ft.Column([history_table], scroll=ft.ScrollMode.AUTO),
                     border=ft.border.all(1, ft.Colors.GREY_200), border_radius=10, expand=True)
    ], expand=True)

    tabs_control = ft.Tabs(selected_index=0, animation_duration=300,
                           tabs=[ft.Tab(text="Kết quả", icon=ft.Icons.DASHBOARD, content=tab_result_content),
                                 ft.Tab(text="Lịch sử", icon=ft.Icons.HISTORY,
                                        content=ft.Container(content=tab_history, padding=20))], expand=True,
                           divider_color=ft.Colors.TRANSPARENT, indicator_color=AppColors.Primary,
                           label_color=AppColors.Primary)

    page.add(header, ft.Container(content=ft.Row([ft.Container(content=left_content, padding=40,
                                                               bgcolor=AppColors.LightCard, border_radius=25,
                                                               shadow=ft.BoxShadow(blur_radius=20,
                                                                                   color=ft.Colors.with_opacity(0.08,
                                                                                                                ft.Colors.BLACK)),
                                                               expand=4, margin=ft.margin.only(right=15)),
                                                  ft.Container(content=tabs_control, padding=10,
                                                               bgcolor=AppColors.LightCard, border_radius=25,
                                                               shadow=ft.BoxShadow(blur_radius=20,
                                                                                   color=ft.Colors.with_opacity(0.08,
                                                                                                                ft.Colors.BLACK)),
                                                               expand=6)], expand=True), padding=30,
                                  bgcolor=AppColors.LightBG, expand=True))


if __name__ == "__main__":
    ft.app(target=main)