package ifmo.mmmlpmsw.cloud

import org.telegram.telegrambots.meta.TelegramBotsApi
import org.telegram.telegrambots.meta.exceptions.TelegramApiRequestException
import org.telegram.telegrambots.updatesreceivers.DefaultBotSession


fun main() {
    val telegramBotsApi = TelegramBotsApi(DefaultBotSession::class.java)
    try {
        telegramBotsApi.registerBot(NeuralCatBot())
    } catch (e: TelegramApiRequestException) {
        e.printStackTrace()
    }
}