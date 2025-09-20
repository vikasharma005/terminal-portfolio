import '@commands/help.css';

import type { ComponentCommand } from '@commands';
import useCommandsState from '@commands';
import generateTabs from '@fn/generate-tabs';
import { i18n } from '@locale';
import { useStore } from '@nanostores/preact';
import type { FunctionalComponent } from 'preact';

const helpMessages = i18n('help', {
	alias: 'alias',
	ctrl_l: 'Ctrl + l',
	ctrl_l_desc: 'clear the console',
	tab_ctrl: 'Tab or Ctrl + i',
	tab_ctrl_desc: 'autocomplete the command',
	up_arrow: 'Up arrow',
	up_arrow_desc: 'show previous command',
});

const descriptionMessages = i18n('commands', {
	about: 'about Vikas Sharma',
	achievements: 'show my achievements and recognition',
	challenge: 'interactive coding challenges',
	clear: 'clear the console',
	education: 'show my educational background',
	contact: 'get my contact information',
	exit: 'exit the terminal',
	experience: 'show my professional experience',
	help: 'list of available commands',
	history: 'show command history',
	projects: 'show all my projects',
	resume: 'go to my resume in web format',
	skills: 'show my technical skills',
	stats: 'show GitHub statistics',
	themes: 'show available themes',
	volunteering: 'show my volunteering experience',
	welcome: 'show welcome message',
});

const MAX_COLUMNS_COMMANDS = 14;
const MAX_COLUMNS_AUTOCOMPLETE = MAX_COLUMNS_COMMANDS + 5;

const Help: FunctionalComponent = () => {
	const h = useStore(helpMessages);
	const d = useStore(descriptionMessages);
	const { list } = useCommandsState();

	return (
		<div
			className='terminal-line-history sm'
			data-testid='help'>
			{list.map(({ alias, command }) => (
				<>
					<div key={command}>
						<span className='command-name'>{command}</span>
						{generateTabs(MAX_COLUMNS_COMMANDS - command.length)}
						{/* @ts-expect-error TS7053: Element implicitly has an any type because expression of type string can't be used to index type */}
						<span className='command-description'>- {d[command]}</span>
					</div>
					{alias ? (
						<div key={`${command}-alias`}>
							{generateTabs(MAX_COLUMNS_COMMANDS + 2)}
							<span>{h.alias}: </span>
							{alias.map((a, index, array) => (
								<>
									<span class='command-name'>{a}</span>
									{index === array.length - 1 ? undefined : ', '}
								</>
							))}
						</div>
					) : undefined}
				</>
			))}
			<div className='autocomplete font-sm text-200 first'>
				{h.tab_ctrl}
				{generateTabs(MAX_COLUMNS_AUTOCOMPLETE - h.tab_ctrl.length)}-&nbsp;
				{h.tab_ctrl_desc}
			</div>
			<div className='autocomplete font-sm text-200'>
				{h.up_arrow}
				{generateTabs(MAX_COLUMNS_AUTOCOMPLETE - h.up_arrow.length)}-&nbsp;
				{h.up_arrow_desc}
			</div>
			<div className='autocomplete font-sm text-200'>
				{h.ctrl_l}
				{generateTabs(MAX_COLUMNS_AUTOCOMPLETE - h.ctrl_l.length)}-&nbsp;
				{h.ctrl_l_desc}
			</div>
		</div>
	);
};

const HelpCommand: ComponentCommand = {
	command: 'help',
	component: Help,
};

export default HelpCommand;
