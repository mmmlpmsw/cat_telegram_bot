package ifmo.mmmlpmsw.cloud.bot

import org.telegram.telegrambots.meta.TelegramBotsApi
import org.telegram.telegrambots.meta.exceptions.TelegramApiRequestException
import org.telegram.telegrambots.updatesreceivers.DefaultBotSession


fun main() {
    try {
        TelegramBotsApi(DefaultBotSession::class.java).registerBot(NeuralCatBot())
    } catch (e: TelegramApiRequestException) {
        e.printStackTrace()
    }
}
