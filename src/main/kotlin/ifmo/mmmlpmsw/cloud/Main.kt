package ifmo.mmmlpmsw.cloud

import org.telegram.telegrambots.ApiContextInitializer
import org.telegram.telegrambots.TelegramBotsApi
import org.telegram.telegrambots.exceptions.TelegramApiRequestException


fun main() {
    ApiContextInitializer.init()
    val telegramBotsApi = TelegramBotsApi()

    try {
        telegramBotsApi.registerBot(NeuralCatBot())
    } catch (e: TelegramApiRequestException) {
        e.printStackTrace()
    }
}