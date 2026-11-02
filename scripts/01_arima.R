# =============================================================================
# 01_arima_v2.R
# ARIMA con regressori di Fourier + dummy festività australiane
# Serie: conteggio orario pedoni, Australia, 2015–2020
# Forecast target: 1439 ore → Gen 2020 – Feb 2020
# =============================================================================
# MODIFICHE rispetto a v1:
#   - d=0 forzato (ADF confermava stazionarietà; d=1 ancorava il forecast
#     ai valori minimi di dicembre impedendo la risalita stagionale)
#   - K fissati dai risultati della grid search precedente (K_daily=12,
#     K_weekly=6, K_annual=1 → AICc=128158.73) → grid search saltata
#   - stepwise=TRUE, approximation=TRUE per contenere i tempi (~20 min)
# =============================================================================
#
# DIPENDENZE: student_dataset.csv nella working directory
#             (oppure series_objects.rds generato da 00_exploratory.R)
#
# OUTPUT:
#   - arima_predictions.csv  → colonne: time, ARIMA
#   - arima_model.rds        → oggetto modello per report
# =============================================================================

# --- 0. PACCHETTI ------------------------------------------------------------
library(tidyverse)
library(lubridate)
library(forecast)
library(readr)

set.seed(42)

# --- 1. CARICAMENTO DATI -----------------------------------------------------
cat("=== 1. CARICAMENTO DATI ===\n")

if (file.exists("series_objects.rds")) {
  objs    <- readRDS("series_objects.rds")
  df_obs  <- objs$df_obs
  df_fore <- objs$df_fore
  cat("  ✓ Caricato da series_objects.rds\n")
} else {
  df_raw <- read_csv("student_dataset.csv",
                     col_types = cols(time  = col_character(),
                                      value = col_double()))
  df_raw  <- df_raw %>% mutate(time = ymd_hms(time, tz = "UTC")) %>% arrange(time)
  df_obs  <- df_raw %>% filter(!is.na(value))
  df_fore <- df_raw %>% filter(is.na(value))
  cat("  ✓ Caricato da student_dataset.csv\n")
}

N <- nrow(df_obs)
h <- nrow(df_fore)

cat(sprintf("  Training:  %d osservazioni  (%s → %s)\n",
            N, format(min(df_obs$time)), format(max(df_obs$time))))
cat(sprintf("  Forecast:  %d ore  (%s → %s)\n\n",
            h, format(min(df_fore$time)), format(max(df_fore$time))))


# --- 2. TRASFORMAZIONE BOX-COX -----------------------------------------------
# skewness=1.144, CV=101% → stabilizzazione della varianza necessaria.
# Offset=0.5 per gestire gli zeri.
# -----------------------------------------------------------------------------
cat("=== 2. TRASFORMAZIONE BOX-COX ===\n")

OFFSET <- 0.5
y_raw  <- df_obs$value + OFFSET

lambda <- BoxCox.lambda(y_raw, method = "guerrero", lower = 0, upper = 1)
cat(sprintf("  Lambda stimato (Guerrero): %.4f\n", lambda))
cat(sprintf("  Interpretazione: %s\n\n",
            ifelse(lambda < 0.1, "quasi-logaritmica",
                   ifelse(lambda < 0.6, "radice con potenza intermedia", "quasi-lineare"))))

y_bc <- BoxCox(y_raw, lambda = lambda)

y_ts_24   <- ts(y_bc, frequency = 24)
y_ts_168  <- ts(y_bc, frequency = 24 * 7)
y_ts_8766 <- ts(y_bc, frequency = 24 * 365.25)


# --- 3. DUMMY VARIABILI FESTIVITÀ --------------------------------------------
cat("=== 3. DUMMY VARIABILI FESTIVITÀ ===\n")

build_holidays <- function(dates) {
  as.matrix(tibble(
    christmas     = as.numeric(month(dates) == 12 & day(dates) %in% 24:26),
    new_year      = as.numeric(
      (month(dates) == 12 & day(dates) == 31) |
        (month(dates) ==  1 & day(dates) %in%  1:2)),
    australia_day = as.numeric(month(dates) == 1 & day(dates) == 26),
    anzac_day     = as.numeric(month(dates) == 4 & day(dates) == 25),
    weekend       = as.numeric(wday(dates) %in% c(1, 7))
  ))
}

hol_train <- build_holidays(df_obs$time)
hol_fore  <- build_holidays(df_fore$time)

cat("  Festività in training:\n")
cat(sprintf("    Christmas:     %4d ore\n", sum(hol_train[,"christmas"])))
cat(sprintf("    New Year:      %4d ore\n", sum(hol_train[,"new_year"])))
cat(sprintf("    Australia Day: %4d ore\n", sum(hol_train[,"australia_day"])))
cat(sprintf("    ANZAC Day:     %4d ore\n", sum(hol_train[,"anzac_day"])))
cat(sprintf("    Weekend:       %4d ore\n", sum(hol_train[,"weekend"])))
cat("\n  Festività nel periodo di FORECAST:\n")
cat(sprintf("    New Year:      %4d ore\n", sum(hol_fore[,"new_year"])))
cat(sprintf("    Australia Day: %4d ore\n", sum(hol_fore[,"australia_day"])))
cat("\n")


# --- 4. PARAMETRI K ----------------------------------------------------------
# Risultati grid search precedente (18 combinazioni testate):
#   K_daily=12, K_weekly=6, K_annual=1 → AICc = 128158.73  ← MIGLIORE
#   K_daily=10, K_weekly=6, K_annual=2 → AICc = 128247.12
#   K_daily=12, K_weekly=6, K_annual=2 → AICc = 128308.34
# Grid search saltata per risparmiare tempo.
# -----------------------------------------------------------------------------
cat("=== 4. PARAMETRI K (da grid search precedente) ===\n")

K_DAILY  <- 12
K_WEEKLY <- 6
K_ANNUAL <- 1

cat(sprintf("  K_daily=%d  K_weekly=%d  K_annual=%d  (AICc grid=128158.73)\n\n",
            K_DAILY, K_WEEKLY, K_ANNUAL))


# --- 5. STIMA MODELLO FINALE -------------------------------------------------
# CORREZIONE CHIAVE: d=0 forzato.
# Motivazione: con d=1 auto.arima differenziava la serie e il forecast
# si ancorava agli ultimi valori osservati (dicembre, minimo stagionale),
# producendo previsioni ~17 invece dei ~100 attesi per Jan-Feb.
# L'ADF test dell'EDA confermava stazionarietà (p<0.01), quindi d=0
# è teoricamente corretto e risolve il problema di ancoraggio.
# -----------------------------------------------------------------------------
cat("=== 5. STIMA MODELLO FINALE ===\n")
cat("  d=0 forzato | stepwise=TRUE | approximation=TRUE\n")
cat("  Attesa stimata: 15-25 minuti\n\n")

xreg_train_final <- cbind(
  fourier(y_ts_24,   K = K_DAILY),
  fourier(y_ts_168,  K = K_WEEKLY),
  fourier(y_ts_8766, K = K_ANNUAL),
  hol_train
)

cat(sprintf("  Regressori totali: %d\n", ncol(xreg_train_final)))
cat(sprintf("    Fourier giornaliero (%d K × 2): %d\n",  K_DAILY,  2*K_DAILY))
cat(sprintf("    Fourier settimanale (%d K × 2): %d\n",  K_WEEKLY, 2*K_WEEKLY))
cat(sprintf("    Fourier annuale (%d K × 2):     %d\n",  K_ANNUAL, 2*K_ANNUAL))
cat(sprintf("    Dummy festività:                 %d\n", ncol(hol_train)))
cat("\n")

fit_arima <- auto.arima(
  y_ts_24,
  xreg          = xreg_train_final,
  seasonal      = FALSE,   # stagionalità gestita dai Fourier
  d             = 0,       # ← CORREZIONE: forza d=0
  stepwise      = TRUE,    # greedy ma veloce (~20 min)
  approximation = TRUE,    # AICc approssimato per velocità
  trace         = TRUE
)

cat("\n\n")
cat("  ✓ Modello selezionato:", fit_arima$method, "\n")
cat("  AICc:", round(fit_arima$aicc, 4), "\n")
cat("  BIC: ", round(fit_arima$bic,  4), "\n\n")

cat("--- Summary completo ---\n")
print(summary(fit_arima))


# --- 6. DIAGNOSTICA RESIDUI --------------------------------------------------
cat("\n=== 6. DIAGNOSTICA RESIDUI ===\n")

checkresiduals(fit_arima, lag = 48, plot = TRUE)

resid_arima <- residuals(fit_arima)

mae_is  <- mean(abs(resid_arima))
rmse_is <- sqrt(mean(resid_arima^2))
cat(sprintf("  MAE  in-sample (scala BC): %.4f\n", mae_is))
cat(sprintf("  RMSE in-sample (scala BC): %.4f\n", rmse_is))

fitdf <- sum(fit_arima$arma[1:2])
lb_24 <- Box.test(resid_arima, lag = 24, type = "Ljung-Box", fitdf = fitdf)
lb_48 <- Box.test(resid_arima, lag = 48, type = "Ljung-Box", fitdf = fitdf)
cat(sprintf("  Ljung-Box p-value (lag=24): %.4f  →  %s\n",
            lb_24$p.value,
            ifelse(lb_24$p.value > 0.05, "✓ residui non autocorrelati", "⚠ autocorrelazione residua")))
cat(sprintf("  Ljung-Box p-value (lag=48): %.4f  →  %s\n\n",
            lb_48$p.value,
            ifelse(lb_48$p.value > 0.05, "✓ residui non autocorrelati", "⚠ autocorrelazione residua")))

par(mfrow = c(2, 1))
Acf(resid_arima,  lag.max = 168, main = "ACF Residui ARIMA")
Pacf(resid_arima, lag.max = 168, main = "PACF Residui ARIMA")
par(mfrow = c(1, 1))


# --- 7. FORECAST -------------------------------------------------------------
cat("=== 7. FORECAST ===\n")

xreg_future <- cbind(
  fourier(y_ts_24,   K = K_DAILY,  h = h),
  fourier(y_ts_168,  K = K_WEEKLY, h = h),
  fourier(y_ts_8766, K = K_ANNUAL, h = h),
  hol_fore
)

fc <- forecast(fit_arima, xreg = xreg_future, h = h)

# Back-transformation: InvBoxCox → rimuovi offset → clip a 0
fc_point <- pmax(InvBoxCox(as.numeric(fc$mean),       lambda) - OFFSET, 0)
fc_lo95  <- pmax(InvBoxCox(as.numeric(fc$lower[,2]),  lambda) - OFFSET, 0)
fc_hi95  <- pmax(InvBoxCox(as.numeric(fc$upper[,2]),  lambda) - OFFSET, 0)
fc_lo80  <- pmax(InvBoxCox(as.numeric(fc$lower[,1]),  lambda) - OFFSET, 0)
fc_hi80  <- pmax(InvBoxCox(as.numeric(fc$upper[,1]),  lambda) - OFFSET, 0)

cat("  ✓ Forecast completato\n")
cat("  Statistiche previsioni (scala originale):\n")
print(summary(fc_point))
cat(sprintf("  Valori negativi clippati a 0: %d\n\n",
            sum(InvBoxCox(as.numeric(fc$mean), lambda) - OFFSET < 0)))

# Controllo di sanità: media forecast vs media storica Jan-Feb
media_storica_janfeb <- df_obs %>%
  filter(month(time) %in% c(1, 2)) %>%
  pull(value) %>% mean()
cat(sprintf("  Media storica Jan-Feb (2015-2019): %.1f\n", media_storica_janfeb))
cat(sprintf("  Media forecast Jan-Feb 2020:       %.1f\n", mean(fc_point)))
cat(sprintf("  Rapporto forecast/storico:         %.2f  (atteso vicino a 1)\n\n",
            mean(fc_point) / media_storica_janfeb))


# --- 8. VISUALIZZAZIONI ------------------------------------------------------
cat("=== 8. VISUALIZZAZIONI ===\n")

df_fc <- tibble(
  time = df_fore$time,
  fc   = fc_point,
  lo80 = fc_lo80,
  hi80 = fc_hi80,
  lo95 = fc_lo95,
  hi95 = fc_hi95
)

# Plot 1: Ultimi 30 giorni osservati + forecast
last_30 <- df_obs %>% tail(30 * 24)

p1 <- ggplot() +
  geom_ribbon(data = df_fc, aes(x = time, ymin = lo95, ymax = hi95),
              fill = "lightblue", alpha = 0.3) +
  geom_ribbon(data = df_fc, aes(x = time, ymin = lo80, ymax = hi80),
              fill = "steelblue", alpha = 0.3) +
  geom_line(data = last_30, aes(x = time, y = value),
            color = "black", linewidth = 0.5) +
  geom_line(data = df_fc, aes(x = time, y = fc),
            color = "steelblue", linewidth = 0.8) +
  geom_vline(xintercept = max(df_obs$time),
             linetype = "dashed", color = "red") +
  labs(
    title    = "ARIMA + Fourier: Forecast Gen–Feb 2020",
    subtitle = paste0("Modello: ", fit_arima$method,
                      "  |  d=0  |  K=c(", K_DAILY, ",", K_WEEKLY, ",", K_ANNUAL, ")",
                      "  |  Bande 80% e 95%"),
    x = "Tempo (UTC)", y = "Conteggio pedoni"
  ) +
  theme_minimal(base_size = 12)
print(p1)

# Plot 2: Profilo orario medio — storico Jan-Feb vs forecast
hist_janfeb <- df_obs %>%
  filter(month(time) %in% c(1, 2)) %>%
  mutate(hour = hour(time)) %>%
  group_by(hour) %>%
  summarise(media_storica = mean(value), .groups = "drop")

fc_hourly <- df_fc %>%
  mutate(hour = hour(time)) %>%
  group_by(hour) %>%
  summarise(media_forecast = mean(fc), .groups = "drop")

p2 <- left_join(hist_janfeb, fc_hourly, by = "hour") %>%
  pivot_longer(-hour, names_to = "serie", values_to = "valore") %>%
  ggplot(aes(x = hour, y = valore, color = serie)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  scale_color_manual(
    values = c("media_storica" = "gray50", "media_forecast" = "steelblue"),
    labels = c("Storico Jan–Feb 2015–2019", "Forecast Jan–Feb 2020")
  ) +
  scale_x_continuous(breaks = 0:23) +
  labs(title = "Profilo Orario Medio: Storico vs Forecast",
       x = "Ora del giorno", y = "Conteggio medio", color = "") +
  theme_minimal(base_size = 12)
print(p2)

# Plot 3: Zoom prime 2 settimane
p3 <- df_fc %>%
  filter(time <= min(time) + days(14)) %>%
  ggplot(aes(x = time)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = "lightblue", alpha = 0.3) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), fill = "steelblue", alpha = 0.3) +
  geom_line(aes(y = fc), color = "steelblue", linewidth = 0.8) +
  labs(title    = "Zoom: Prime 2 Settimane di Forecast",
       subtitle = "1–14 Gennaio 2020  |  Include Capodanno (1 gen)",
       x = "Tempo", y = "Conteggio pedoni") +
  theme_minimal(base_size = 12)
print(p3)

# Plot 4: Confronto anni storici Gen-Feb allineati
df_obs_janfeb <- df_obs %>%
  filter(month(time) %in% c(1, 2)) %>%
  mutate(year           = as.character(year(time)),
         reference_time = make_datetime(2000, month(time), day(time), hour(time)))

df_fc_aligned <- df_fc %>%
  mutate(year           = "Forecast 2020",
         reference_time = make_datetime(2000, month(time), day(time), hour(time)))

p4 <- ggplot() +
  geom_line(data = df_obs_janfeb,
            aes(x = reference_time, y = value, group = year, color = year),
            linetype = "dashed", linewidth = 0.5, alpha = 0.7) +
  geom_line(data = df_fc_aligned,
            aes(x = reference_time, y = fc),
            color = "black", linewidth = 1.2) +
  scale_color_brewer(palette = "Set2") +
  labs(title    = "Confronto: Pattern Gen-Feb Storico vs Forecast",
       subtitle = "Tratteggiate colorate = storico per anno  |  Nera = forecast 2020",
       x = "Tempo normalizzato", y = "Conteggio pedoni", color = "Anno") +
  theme_minimal(base_size = 12)
print(p4)


# --- 9. SALVATAGGIO ----------------------------------------------------------
# --- 9. SALVATAGGIO ----------------------------------------------------------
cat("\n=== 9. SALVATAGGIO RISULTATI ===\n")

df_arima_out <- tibble(
  time   = df_fore$time,
  value  = df_fore$value,        # NA (valori mancanti del dataset originale)
  ARIMA  = round(fc_point, 4)
)

write_csv(df_arima_out, "arima_predictions.csv")
cat(sprintf("  ✓ arima_predictions.csv  (%d righe × 3 colonne)\n", nrow(df_arima_out)))
saveRDS(
  list(fit_arima  = fit_arima,
       lambda     = lambda,
       OFFSET     = OFFSET,
       K          = c(K_DAILY, K_WEEKLY, K_ANNUAL),
       y_ts_24    = y_ts_24,
       y_ts_168   = y_ts_168,
       y_ts_8766  = y_ts_8766,
       xreg_train = xreg_train_final,
       df_fc      = df_fc),
  file = "arima_model.rds"
)
cat("  ✓ arima_model.rds\n\n")

cat("  Prime 10 previsioni:\n")
print(head(df_arima_out, 10))
cat("  Ultime 5 previsioni:\n")
print(tail(df_arima_out, 5))


# --- 10. RIEPILOGO FINALE ----------------------------------------------------
cat("\n==========================================\n")
cat("           RIEPILOGO FINALE\n")
cat("==========================================\n")
cat("  Modello:           ARIMA(p,0,q) con Fourier\n")
cat("  Trasformazione:    Box-Cox (lambda =", round(lambda, 4), ")\n")
cat("  d forzato:         0  (ADF stazionario; corregge ancoraggio)\n")
cat("  Fourier K:         daily=", K_DAILY, "| weekly=", K_WEEKLY, "| annual=", K_ANNUAL, "\n")
cat("  Regressori tot.:  ", ncol(xreg_train_final), "\n")
cat("  Festività:         christmas, new_year, australia_day, anzac_day, weekend\n")
cat("  AICc:             ", round(fit_arima$aicc, 2), "\n")
cat("  BIC:              ", round(fit_arima$bic,  2), "\n")
cat("  MAE in-sample:    ", round(mae_is, 4), " (scala Box-Cox)\n")
cat("  Ore previste:     ", h, "\n")
cat("  Range forecast:   ", round(min(fc_point), 1), "–", round(max(fc_point), 1), "\n")
cat("  Media forecast:   ", round(mean(fc_point), 1),
    " (storico Jan-Feb:", round(media_storica_janfeb, 1), ")\n")
cat("==========================================\n")
cat("  ✓ COMPLETATO!\n")
cat("==========================================\n")
